from error_probs_model import ErrorProbsModel
import numpy as np

def select_fixed_annotator(instances, **kwargs):

	annotators = np.zeros(len(instances), dtype=int) + kwargs['annotator']

	return annotators

def select_random_annotator(instances, **kwargs):
	
	selected_idx = instances
	dataset = kwargs['dataset']
	n_annotators = dataset.y_DL.shape[1]
	n_instances = selected_idx.shape[0]

	annotators = []
	selected_y = dataset.y_DL[selected_idx]
	if 0 in np.sum(np.isnan(selected_y), axis = 1):
		print("hhh")
	for i in range(n_instances):
		annotators.append(np.random.choice(np.where(np.isnan(selected_y[i]))[0]))
		
	return annotators

def select_epm_annotator(instances, **kwargs):

	selected_idx = instances
	dataset = kwargs['dataset']

	X_train = dataset.X_train

	y_DL = dataset.y_DL
	epm = ErrorProbsModel(len(np.unique(dataset.y_train)))
	epm.fit(X_train, y_DL)
	label_accuracy = epm.predict(X_train[selected_idx])

	# prevent same annotator from labeling again
	labels = y_DL[selected_idx]
	label_accuracy[~np.isnan(labels)] = 0.

	annotators = np.argmax(label_accuracy, axis=1)
	return annotators