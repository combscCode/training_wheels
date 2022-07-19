# Similar Datasets

SimilarDatasetsWarning should be raised in situations where the user is in danger of
overfitting a set of datapoints. It can be raised when the exact same dataset is used
to train and test a model, but it can also be raised when the testing and training
datasets are too similar to one another.

Training and testing on similar datasets is dangerous because it leads to
overoptimistic predictions of the model's performance. A model should be tested using
novel data in order to properly simulate what will happen when the model is deployed
in a real-world scenario where it has to make predictions on novel inputs.


Example:
```python
from sklearn import svm
import numpy as np

X = np.array([1,2,3,4])
y = [2,4,6,8]
clf = svm.SVC()
clf.fit(X,y)

clf.predict(X) # This should raise a UserWarning
```
