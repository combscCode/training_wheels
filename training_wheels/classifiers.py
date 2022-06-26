import warnings

from training_wheels.compare_data import DataComparer
from training_wheels.exceptions import SimilarDatasetsWarning

"""Obviously not finished, just making
something that reflects the ultimate goal."""
class GuardedClassifier():
	def __init__(self, model):
		self.model = model

	def fit(self, X, y):
		self.comparer = DataComparer(X)
		self.model.fit(X, y)

	def predict(self, X):
	"""What do we do when predicting single datapoints
	rather than being fed a matrix?"""
		if self.comparer.is_same(X):
			msg = "Attempting to predict the dataset you trained on."
			warnings.warn(msg, SimilarDatasetsWarning)
		return self.model.predict(X, y)
