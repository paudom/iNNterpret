from __future__ import absolute_import

# -- IMPORT -- #
from ..utils.interfaces import Method

class GlobalSurrogate(Method):
	"""CLASS::GlobalSurrogate:
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

	def __repr__(self):
		return super().__repr__()+self.__class__.__name__+'>'