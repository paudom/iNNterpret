from __future__ import absolute_import

# -- IMPORT -- #

class SHAPValues():
	"""CLASS::ShapValues:
		---
		Description:
		---
		Computes the relevance of each feature separately
		Arguments:
		---
		Link:
		>- https://arxiv.org/abs/1705.07874."""
	def __init__(self):
		raise NotImplementedError

	def interpret(self):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			Returns:
			---"""
		pass
	
	def __repr__(self):
		return super().__repr__()+self.__class__.__name__+'>'