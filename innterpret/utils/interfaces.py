from __future__ import absolute_import

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
		#if not hasattr(cls,'run'):
			#raise NotImplementedError
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
		#if not hasattr(cls,'up'):
			#raise NotImplementedError
		#if not hasattr(cls,'down'):
		   #raise NotImplementedError
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
		#if not hasattr(cls,'interpret'):
			#raise NotImplementedError
		return super(Method,cls).__new__(cls)