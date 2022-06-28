import warnings

from training_wheels.compare_data import DatasetComparer
from training_wheels.exceptions import SimilarDatasetsWarning

class GuardedPredictor():
  """Obviously not finished, just making
  something that reflects the ultimate goal."""
  def __init__(self, model):
    self.model = model

  def fit(self, X, y, **params):
    self.comparer = DatasetComparer(X)
    self.model.fit(X, y, **params)

  def predict(self, X, **params):
    """What do we do when predicting single datapoints
    rather than being fed a matrix?"""
    if self.comparer.is_same(X):
      msg = "Attempting to predict the dataset you trained on."
      warnings.warn(msg, SimilarDatasetsWarning)
    return self.model.predict(X, **params)

  def __getattr__(self, attr):
    if attr in self.__dict__:
      return getattr(self, attr)
    return getattr(self.model, attr)

