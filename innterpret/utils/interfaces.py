from __future__ import absolute_import

# -- IMPORT -- #
from .exceptions import InterfaceException

# -- INTERFACES -- #
class Rule(object):
	"""INTERFACE::Rule:
		---
		Description:
		---
		>Rule interface.
		Required:
		---
		>- METHOD::RUN"""
	def __new__(cls,*args,**kwargs):
		if not hasattr(cls,'run'):
			raise InterfaceException(cls.__class__.__name__+' needs to implement the method "run".')
		return super(Rule,cls).__new__(cls)

class DeconvLayer(object):
	"""INTERFACE::DeconvLayer:
		---
		Description:
		---
		> Deconvolution Layer Interface.
		Required:
		---
		>- METHOD::UP.
		>- METHOD::DOWN."""
	def __new__(cls,*args,**kwargs):
		if not hasattr(cls,'up'):
			raise InterfaceException(cls.__class__.__name__+' needs to implement the method "up".')
		if not hasattr(cls,'down'):
			raise InterfaceException(cls.__class__.__name__+' needs to implement the method "down".')
		return super(DeconvLayer,cls).__new__(cls)

class Method(object):
	"""INTERFACE::Method:
		---
		Description:
		---
		> Method Interface
		Required:
		---
		>- METHOD::INTERPRET."""
	def __new__(cls,*args,**kwargs):
		if not hasattr(cls,'interpret'):
			raise InterfaceException(cls.__class__.__name__+' needs to implement the method "interpret".')
		return super(Method,cls).__new__(cls)