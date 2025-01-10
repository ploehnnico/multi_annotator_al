import numpy as np
from annotlib.cluster_based import ClusterBasedAnnot

def simulate_annotators(labels, n_annotators=5, seed=40):
	np.random.seed(seed)
	random_state = np.random.RandomState(seed)
	y_true = labels
	X_trans = np.zeros((len(y_true),2))
	n_classes = len(np.unique(y_true))
	res = {}
	for s in ['o', 'y']:

		if s == 'x':
			pass

		if s == 'y':
			A = random_state.uniform(1/n_classes, 1, size=n_annotators*n_classes).reshape((n_annotators, n_classes))
			C = np.empty((n_annotators, n_classes, 2))
			C[:, :, 0] = A
			C[:, :, 1] = A
			annot = ClusterBasedAnnot(X=X_trans, y_true=y_true, y_cluster=y_true, n_annotators=n_annotators,
                                      cluster_labelling_acc=C, random_state=6)

		elif s == 'o':
            # simulate annotators with uniform performance values
            #n_annotators = np.random.choice([4, 5, 6])
			y_cluster_const = np.zeros(len(X_trans), dtype=int)
			min_label_acc = 1. / n_classes
			label_acc_step = (0.9 - min_label_acc) / (n_annotators + 1)
			mean_label_acc = np.linspace(min_label_acc, 0.9 - 2 * label_acc_step, n_annotators)
			C = np.empty((n_annotators, 1, 2))
			for a in range(n_annotators):
				v = np.random.uniform(mean_label_acc[a], mean_label_acc[a] + 2 * label_acc_step)
				C[a, :, :] = v	
			annot = ClusterBasedAnnot(X=X_trans, y_true=y_true, y_cluster=y_cluster_const, n_annotators=n_annotators,
                                      cluster_labelling_acc=C, random_state=6)

		res[s] = annot.Y_
	return res