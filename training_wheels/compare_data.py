"""Functions used for comparing datasets."""

"""This should be used to check similarity
between various datasets. Training and testing
on the same data is a prevalent issue for beginners
and this should be used to avoid this."""
class DatasetComparer():
	
	def __init__(self, X):
		self.X = X

	def is_same(self, Y):
		return self.X is Y

	
