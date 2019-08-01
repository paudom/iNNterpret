from __future__ import absolute_import

# -- IMPORT -- #
from ..utils.interfaces import Method

class LIMEModel(Method):
	"""CLASS::LIMEModel:
		---
		Description:
		---
		Local approximations using linear models or decision trees.
		Arguments:
		---
		Link:
		---
		>- http://arxiv.org/abs/1602.04938."""
	def __init__(self):
		raise NotImplementedError
	
	def interpret(self):
		"""METHOD::INTERPRET:
			---
			Raises:
			>- NotImplementedError."""
		raise NotImplementedError
	
	def __repr__(self):
		return super().__repr__()+'Local Interpretable Model Explanations (LIME)>'