"""
The :mod:`training_wheels.exceptions` module contains all
warnings and exceptions used by Training Wheels.
"""

class SimilarDatasetsWarning(UserWarning):
	"""Warning class to raise when datasets are too similar.

	
	"""

class PerformanceWarning(UserWarning):
	"""Warning class to raise when a model's performance is suspiciously high.

	"""
