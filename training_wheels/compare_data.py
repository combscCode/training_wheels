"""Functions used for comparing datasets."""

"""This should be used to check similarity
between various datasets. Training and testing
on the same data is a prevalent issue for beginners
and this should be used to avoid this."""

import warnings

from training_wheels.config import DATASET_SIMILARITY_THRESHOLD
from training_wheels.exceptions import SimilarDatasetsWarning

class DatasetComparer():


  def __init__(self, X):
    self.X = X
    self.X_set = set()
    self.seen_items = 0
    for item in X:
        # TODO: I'll need to catch exceptions here if the
        # item isn't hashable.
        self.X_set.add(frozenset(item))

  def record_and_check_item(self, x):
    if frozenset(x) in self.X_set:
      self.seen_items += 1
      if self.seen_items / len(self.X) > DATASET_SIMILARITY_THRESHOLD:
          msg = "You're trying to run predictions on datapoints that have already been seen during training."
          warnings.warn(msg, SimilarDatasetsWarning)

  def is_same(self, Y) -> bool:
    return self.X is Y

  def is_unbalanced(self) -> bool:
    pass
    # TODO: Make functions that will let user know if
    # they are splitting training/testing in a weird way.
