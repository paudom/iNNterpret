from __future__ import absolute_import

# -- IMPORT -- #
from tensorflow.python.ops import nn_ops, gen_nn_ops
import keras.backend as K
import tensorflow as tf
import copy as cp
import numpy as np

class ZPlus(object):
	""">> CLASS:ZPLUS: Rule to compute the relevance propagation."""
	def __init__(self,currLayer,nextAct):
		self.layer = currLayer
		self.name = currLayer.name
		self.type = self.layer.__class__.__name__
		self.act = K.identity(nextAct)
		self.maxValue = K.epsilon()
		self.minValue = -K.epsilon()
		
	def run(self,R,ignoreBias=False):
		""">> RUN: run the relevance propagation for the current layer."""
		if self.type == 'Dense':
			return self.run_dense(R,ignoreBias)
		elif self.type == 'MaxPooling2D':
			return self.run_pool(R)
		elif self.type == 'Conv2D':
			return self.run_conv(R,ignoreBias)
		elif self.type == 'Flatten':
			return self.run_flatten(R)
		else:
			raise NotImplementedError
			
	def run_dense(self,R,ignoreBias=False):
		""">> RUN_DENSE: Run the relevance propagation for dense layers."""
		weights = self.layer.get_weights()
		self.W = K.maximum(weights[0],0.)
		self.B = K.maximum(weights[1],0.)
		Z = K.dot(self.act,self.W)+K.epsilon()
		if not ignoreBias:
			Z += self.B
		S = R/Z
		C = K.dot(S,K.transpose(self.W))
		return K.clip(self.act*C,self.minValue,self.maxValue)
	
	def run_flatten(self,R):
		""">> RUN_FLATTEN: Run the relevance propagation for flatten layers."""
		shape = self.act.get_shape().as_list()
		shape[0] = -1
		return K.reshape(R, shape)
	
	def run_pool(self,R):
		""">> RUN_POOL: Run the relevance propagation for pool layers."""
		poolSize = (1,self.layer.pool_size[0],self.layer.pool_size[1],1)
		strdSize = (1,self.layer.strides[0],self.layer.strides[1],1)
		pooled = tf.nn.max_pool(self.act, 
						  ksize = poolSize, 
						  strides = strdSize, 
						  padding = self.layer.padding.upper())
		Z = K.maximum(pooled,0.)+K.epsilon()
		S = R/Z
		C = gen_nn_ops.max_pool_grad_v2(self.act,
										Z, S, poolSize, strdSize,
										padding = self.layer.padding.upper())
		return K.clip(self.act*C,self.minValue,self.maxValue)
	
	def run_conv(self,R,ignoreBias):
		""">> RUN_CONV: Run the relevance propagation for conv layers."""
		strdSize = (1,self.layer.strides[0],self.layer.strides[1],1)
		weights = self.layer.get_weights()
		self.W = K.maximum(weights[0],0.)
		self.B = K.maximum(weights[1],0.)
		Z = tf.nn.conv2d(self.act, 
						 self.W, 
						 strides = strdSize,
						 padding = self.layer.padding.upper())+K.epsilon()
		if not ignoreBias:
			Z += self.B
		S = R/Z
		C = nn_ops.conv2d_backprop_input(K.shape(self.act),
										  self.W,
										  S,strdSize,self.layer.padding.upper())
		return K.clip(self.act*C,self.minValue,self.maxValue)

class ZAlpha(object):
	""">> CLASS:ZALPHA: Rule to compute the relevance propagation."""
	def __init__(self,currLayer,nextAct,alpha):
		self.alpha = alpha
		self.beta = 1-alpha
		self.layer = currLayer
		self.name = currLayer.name
		self.type = self.layer.__class__.__name__
		self.act = K.identity(nextAct)
		self.maxValue = K.epsilon()
		self.minValue = -K.epsilon()
		
	def run(self,R,ignoreBias=False):
		""">> RUN: run the relevance propagation for the current layer."""
		if self.type == 'Dense':
			return self.run_dense(R,ignoreBias)
		elif self.type == 'MaxPooling2D':
			return self.run_pool(R)
		elif self.type == 'Conv2D':
			return self.run_conv(R,ignoreBias)
		elif self.type == 'Flatten':
			return self.run_flatten(R)
		else:
			raise NotImplementedError
		
	def run_dense(self,R,ignoreBias=False):
		""">> RUN_DENSE: Run the relevance propagation for dense layers."""
		weights = self.layer.get_weights()
		self.maxW = K.maximum(weights[0],0.); self.maxB = K.maximum(weights[1],0.)
		self.minW = K.minimum(weights[0],0.); self.minB = K.minimum(weights[1],0.)   
		Za = K.dot(self.act,self.maxW)+K.epsilon(); Zb = K.dot(self.act,self.minW)-K.epsilon()
		if not ignoreBias:
			Za += self.maxB; Zb += self.minB
		Sa = R/Za; Sb = R/Zb
		Ca = K.dot(Sa,K.transpose(self.maxW)); Cb = K.dot(Sb,K.transpose(self.minW))
		Rn = self.act*(self.alpha*Ca+self.beta*Cb)
		return K.clip(Rn,self.minValue,self.maxValue)
		
	def run_flatten(self,R):
		""">> RUN_FLATTEN: Run the relevance propagation for flatten layers."""
		shape = self.act.get_shape().as_list()
		shape[0] = -1
		return K.reshape(R, shape)
	
	def run_pool(self,R):
		""">> RUN_POOL: Run the relevance propagation for pool layers."""
		poolSize = (1,self.layer.pool_size[0],self.layer.pool_size[1],1)
		strdSize = (1,self.layer.strides[0],self.layer.strides[1],1)
		pooled = tf.nn.max_pool(self.act, 
						  ksize = poolSize, 
						  strides = strdSize, 
						  padding = self.layer.padding.upper())
		Za = K.maximum(pooled,0.)+K.epsilon(); Zb = K.minimum(pooled,0.)-K.epsilon()
		Sa = R/Za; Sb = R/Zb
		Ca = gen_nn_ops.max_pool_grad_v2(self.act,
										Za, Sa, poolSize, strdSize,
										padding = self.layer.padding.upper())
		Cb = gen_nn_ops.max_pool_grad_v2(self.act,
										Zb, Sb, poolSize, strdSize,
										padding = self.layer.padding.upper())
		Rn = self.act*(self.alpha*Ca+self.beta*Cb)
		return K.clip(Rn,self.minValue,self.maxValue)
	
	def run_conv(self,R,ignoreBias=True):
		""">> RUN_CONV: Run the relevance propagation for conv layers."""
		strdSize = (1,self.layer.strides[0],self.layer.strides[1],1)
		weights = self.layer.get_weights()
		self.maxW = K.maximum(weights[0],0.); self.maxB = K.maximum(weights[1],0.)
		self.minW = K.minimum(weights[0],0.); self.minB = K.minimum(weights[1],0.)
		Za = tf.nn.conv2d(self.act, 
						  self.maxW, 
						  strides = strdSize,
						  padding = self.layer.padding.upper())+K.epsilon()
		Zb = tf.nn.conv2d(self.act, 
						  self.minW, 
						  strides = strdSize,
						  padding = self.layer.padding.upper())-K.epsilon()
		if not ignoreBias:
			Za += self.maxB; Zb += self.minB
		Sa = R/Za; Sb = R/Zb
		Ca = nn_ops.conv2d_backprop_input(K.shape(self.act),
										  self.maxW,
										  Sa,strdSize,self.layer.padding.upper())
		Cb = nn_ops.conv2d_backprop_input(K.shape(self.act),
										  self.minW,
										  Sb,strdSize,self.layer.padding.upper())
		Rn = self.act*(self.alpha*Ca+self.beta*Cb)
		return K.clip(Rn,self.minValue,self.maxValue)