from training_wheels.predictors import GuardedPredictor
from training_wheels.exceptions import SimilarDatasetsWarning
from sklearn import svm
import numpy as np
import pytest



def test_duplicated_dataset_1():
    X = np.array([1,2,3,4]).reshape(-1, 1)
    y = np.array([1,2,3,4])
    clf = GuardedPredictor(svm.SVC())
    clf.fit(X, y)
    with pytest.warns(SimilarDatasetsWarning):
        clf.predict(X)