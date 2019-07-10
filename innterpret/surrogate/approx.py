from __future__ import absolute_import

# -- IMPORT -- #
from ..utils.interfaces import Method

class GlobalModel(Method):
	"""CLASS::GlobalModel:
		---
		Description:
		---
		> Tries to approximate the model using linear or decision trees.
		Arguments:
		---"""
	def __init__(self):
		raise NotImplementedError
	
	def interpret(self):
		"""METHOD::INTERPRET:
			---
			Raises:
			>- NotImplementedError."""
		raise NotImplementedError