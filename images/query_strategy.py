import numpy as np

from abc import ABC, abstractmethod
from annotator_selection import select_fixed_annotator, select_random_annotator, select_epm_annotator

rng = np.random.default_rng(12345)

class QueryStrategy(ABC):

	def __init__(self, data_set, **kwargs):
		self.data_set = data_set
		self.budget = kwargs.get('budget')
		self.relabel = kwargs.get('relabel', True)
		self.annotator = kwargs.get('annotator', 0)
		self.annotator_selection = kwargs.get('annotator_selection', 'random')

	def select_annotators(self, instances):

		selection = {'fixed' : select_fixed_annotator, 
			         'random' :select_random_annotator, 'epm': select_epm_annotator} 
	
		annotators = selection[self.annotator_selection](instances, dataset=self.data_set, 
												   		 annotator=self.annotator)

		return annotators

	@abstractmethod
	def compute_scores(self):
		pass
	
	def make_initial_selection(self):

		y_train = self.data_set.y_train

		instances_idx = np.array([i for l in [rng.choice(np.where(y_train==c)[0], 0.5*self.budget, 
						 replace=False) for c in np.unique(y_train)] for i in l])
		
		annotators = self.select_annotators(instances_idx)

		self.data_set.update_entries(instances_idx, annotators)

	def make_query(self, **kwargs):
		
		scores = self.compute_scores(**kwargs)
		

		if not self.relabel:
			scores[self.data_set.get_annotated()] = np.inf
		else:
			scores[self.data_set.get_fully_annotated()] = np.inf
		
		instances_idx = np.argsort(scores)[:self.budget]
		annotators = self.select_annotators(instances_idx)

		self.data_set.update_entries(instances_idx, annotators)


class MarginQS(QueryStrategy):

	def compute_scores(self, **kwargs):
		p = kwargs.get('probabilities')
		score = np.diff(np.sort(p, axis=1))[:,-1]

		return score