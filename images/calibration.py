import numpy as np

from sklearn.metrics import brier_score_loss

def get_brier_score(y_test, y_probs):

	return brier_score_loss(y_test, y_probs)	

def get_calibration_errors(y_true, y_probs, n_bins=10):

	edges = np.arange(0,1.1, 0.1)
	idx = np.digitize(y_probs, edges, right=True) -1

	errors = []
	n_probs = []

	for i in range(n_bins):
		mask = (idx == i)
		n = np.sum(mask)
		n_probs.append(n)

		if n > 0:
			observed_freq = np.mean(y_true[mask])
			predicted_prob = np.mean(y_probs[mask])
			errors.append(np.abs(observed_freq - predicted_prob))
		else:
			errors.append(0)

	return np.array(errors), np.array(n_probs)

def get_ece(y_true, y_probs, n_bins=10):

	errors, n_probs = get_calibration_errors(y_true, y_probs, n_bins)

	ece = np.sum(errors * n_probs) / len(y_true)

	return ece

def get_mce(y_true, y_probs, n_bins=10):
	errors, _ = get_calibration_errors(y_true, y_probs, n_bins)

	max_error = np.max(errors)
	
	return max_error