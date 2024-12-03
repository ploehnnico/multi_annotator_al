import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_consistent_length, check_random_state
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

def rand_arg_max(arr, axis=1, random_state=None):
    """
    Returns index of maximum element per given axis. In case of ties, the index is chosen randomly.

    Parameters
    ----------
    arr: array-like
        Array whose maximum elements' indices are determined.
    axis: int
        Indices of maximum elements are determined along this axis.
    random_state: numeric | np.random.RandomState
        Random state for annotator selection.

    Returns
    -------
    max_indices: array-like
        Indices of maximum elements.
    """
    random_state = check_random_state(random_state)
    arr = np.array(arr)
    arr_max = arr.max(axis, keepdims=True)
    tmp = random_state.uniform(low=1, high=2, size=arr.shape) * (arr == arr_max)
    return tmp.argmax(axis)

def compute_vote_vectors(y, c=None, n_unique_votes=None, probabilistic=False):

    # check input class labels
    y = check_array(y, ensure_2d=False, force_all_finite=False, copy=True)
    y = y if y.ndim == 2 else y.reshape((-1, 1))
    is_nan_y = np.isnan(y)
    y[is_nan_y] = 0
    y = y.astype(int)
    n_unique_votes = np.size(np.unique(y), axis=0) if n_unique_votes is None else n_unique_votes

    # check input confidence scores
    c = np.ones_like(y) if c is None else check_array(c, ensure_2d=False, force_all_finite=False, copy=True)
    c = c if c.ndim == 2 else c.reshape((-1, 1))
    check_consistent_length(y, c)
    check_consistent_length(y.T, c.T)
    c[np.logical_and(np.isnan(c), ~is_nan_y)] = 1

    if probabilistic:
        # compute probabilistic votes per class
        n_annotators = np.size(y, axis=1)
        n_samples = np.size(y, axis=0)
        sample_ids = np.arange(n_samples)
        P = np.array([(1 - c) / (n_unique_votes - 1)] * n_unique_votes).reshape(n_annotators, n_samples, n_unique_votes)
        for a in range(n_annotators):
            P[a, sample_ids,  y[:, a]] = c[:, a]
            P[a, is_nan_y[:, a]] = np.nan
        V = np.nansum(P, axis=0)
        #V_sum = V.sum(axis=1, keepdims=True)
        #is_not_zero = (V_sum != 0).flatten()
        #V[is_not_zero] /= V_sum[is_not_zero]
    else:
        # count class labels per class and weight by confidence scores
        c[np.logical_or(np.isnan(c), is_nan_y)] = 0
        y_off = y + np.arange(y.shape[0])[:, None] * n_unique_votes
        V = np.bincount(y_off.ravel(), minlength=y.shape[0] * n_unique_votes, weights=c.ravel())
        V = V.reshape(-1, n_unique_votes)

    return V

class ErrorProbsModel(BaseEstimator):
    """ErrorProbsModel

    The Error Probability Model (ErrorProbsModel) [1] estimates the annotation performances, i.e. label accuracies,
    of multiple annotators per sample. Given several samples and corresponding label vectors of these annotators,
    the majority vote per sample-label-vector-pair is computed. To estimate an annotator's label accuracies for a given
    sample, a logistic regression model trained on the samples labeled by the annotator are used.

    Parameters
    ----------
    metric: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs: dict,
        Any further parameters are passed directly to the metric/kernel function.

    Attributes
    ----------
    metric_: str,
        The metric must a be a valid kernel defined by the function sklearn.metrics.pairwise.pairwise_kernels.
    n_neighbors_: int,
        Number of nearest neighbours. Default is None, which means all available samples are considered.
    random_state: None | int | numpy.random.RandomState
        The random state used for deciding on majority vote labels in case of ties.
    kwargs_: dict,
        Any further parameters are passed directly to the kernel function.
    pwc_list_: array-like, shape (n_annotators)
        For each annotator one fitted Parzen Window Classifier [2] used to estimate the annotation performance.

    References
    ----------
    [1] Huang, S. J., Chen, J. L., Mu, X., & Zhou, Z. H. (2017). Cost-effective Active Learning from Diverse Labelers.
        Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence (IJCAI-17), 1879–1885.
        Melbourne, Australia.
    [2] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, n_classes, random_state=None, **kwargs):
        self.n_classes_ = int(n_classes)
        if self.n_classes_ <= 1:
            raise ValueError("'n_classes' must be an integer greater than one")

        self.random_state_ = check_random_state(random_state)

        self.kwargs_ = kwargs
        self.lr_list_ = None

    def fit(self, X, y, c=None):
        """
        Given the labels of multiple annotators, this method fits annotator models to estimate annotation performances,
        i.e. label accuracies, of these multiple annotators.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.
        y: array-like, shape (n_samples, n_annotators)
            Labels provided by multiple annotators. An entry y[i, j] indicates that the annotator with index j has not
            provided a label for the sample with index i.
        c: array-like, shape (n_samples, n_annotators)
            Weights for the individual labels.
            Default is c[i, j]=1 as weight for the label entry y[i, j].
        """
        # check input parameters
        X = check_array(X)
        y = check_array(y, force_all_finite=False)
        check_consistent_length(X, y)

        # determine number of annotators
        n_annotators = np.size(y, axis=1)

        # flag for labeled entries
        is_labeled = ~np.isnan(y)

        # compute (confidence weighted majority) vote
        V = compute_vote_vectors(y=y, c=c, n_unique_votes=self.n_classes_)
        y_mv = rand_arg_max(arr=V, axis=1, random_state=self.random_state_)

        # fit PWC per annotator
        self.lr_list_ = []
        for a_idx in range(n_annotators):
            is_correct = np.array(np.equal(y_mv[is_labeled[:, a_idx]], y[is_labeled[:, a_idx], a_idx]), dtype=int)
            if len(np.unique(is_correct)) >= 2:
                lr = LogisticRegression(**self.kwargs_)
                self.lr_list_.append(lr.fit(X[is_labeled[:, a_idx]], is_correct))
            else:
                self.lr_list_.append(np.mean(is_correct))

        return self

    def predict(self, X):
        """
        This method estimates the annotation performances, i.e. label accuracies, of the multiple annotators for each
        given sample in X.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix X is the feature matrix representing the samples.

        Returns
        -------
        Y: matrix-like, shape (n_samples, n_annotators)
            Estimate label accuracy for each sample-annotator-pair.
        """
        if self.lr_list_ is None:
            raise NotFittedError("This ErrorProbsModel instance is not fitted yet. Call 'fit' with appropriate "
                                 "arguments before using this estimator.")
        n_annotators = len(self.lr_list_)
        n_samples = len(X)
        Y = np.empty((n_samples, n_annotators))
        for a in range(n_annotators):
            if isinstance(self.lr_list_[a], LogisticRegression):
                Y[:, a] = self.lr_list_[a].predict_proba(X)[:, 0]
            else:
                Y[:, a] = self.lr_list_[a]
        return Y
