import numpy as np
from sklearn.model_selection import train_test_split

class ALDataset:

	def __init__(self, X_train, y_true, y, test_size=0.2):

		self.X_train = X_train
		self.y_true = y_true
		self.y_train = y

		self.n_classes = len(np.unique(y_true))
		self.n_instances = len(y_true)
		#X_train, X_test, y_train_idx, y_test_idx = train_test_split(X, np.arange(len(y_true)), test_size=test_size)

		#self.y_train = y[y_train_idx]
		#self.y_test = y_true[y_test_idx]

		self.y_DL = np.zeros_like(self.y_train)
		self.y_DL[:] = np.nan

	def get_n_annotators(self):

		return self.y_train.shape[1]

	def get_annotated(self):

		annotated = np.sum(~np.isnan(self.y_DL), axis=1) != 0

		return annotated

	def get_fully_annotated(self):

		n_annotators = self.y_DL.shape[1]
		fully_annotated = np.sum(~np.isnan(self.y_DL), axis=1) == n_annotators

		return fully_annotated


	def update_entries(self, idx, annotators):

		mask = np.zeros_like(self.y_DL, dtype=bool)
		mask[idx, annotators] = True

		self.y_DL[mask] = self.y_train[mask]

	def get_train_set(self):
		mask = self.get_annotated()
		X = self.X_train[mask]
		y_DL = self.y_DL[mask]
		y_train = np.zeros((len(y_DL), self.n_classes))

		for i in range(len(y_DL)):
			valid_annotations = y_DL[i, ~np.isnan(y_DL[i, :])].astype(int)
			if len(valid_annotations) > 0:
				counts = np.bincount(valid_annotations, minlength=self.n_classes)
				y_train[i, :] = counts / len(valid_annotations)

		return X, y_train