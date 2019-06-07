from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.tensor import model_remove_sofmax
from ..utils.data import load_image, reduce_channels, deprocess_image, visualize_heatmap
from tensorflow.python.ops import nn_ops, gen_nn_ops
from keras.models import Model
import keras.backend as K
import numpy as np
import tensorflow as tf
import copy as cp

# -- LRP METHOD -- #
class LRPModel():
	def __init__(self,model,verbose=True):
		print_msg(self.__class__.__name__+' Initialization...')
		print_msg('--------------------------')
		reluModel = model_remove_softmax(model)
		_,_,activations,_ = get_model_parameters(model) 
		self.model = Model(inputs=reluModel.input,outputs=activations)
		if verbose:
			self.model.summary()
		self.optionRule = input(print_msg('(ZPlus)-(ZAlpha): ',show=False,option='input'))
		if self.optionRule == 'ZAlpha':
			self.alpha = int(input(print_msg('Select a Value for Alpha: ',show=False,option='input')))
			assert self.alpha >= 1
		self.R = 0; self.outputR = 0; self.rules = 0
		self.numLayers = len(self.model.layers)
		print_msg('========== DONE ==========\n')

	# >> DEFINE_RULES: goes through the model to save the specified rules
	def define_rules(self,imgData,oneHot=False,verbose=True):
		print_msg('Define Rules...')
		print_msg('--------------------------')       
		layerRules = []; self.outputR = self.model.predict(imgData)
		if oneHot:
			neuron = np.argmax(self.outputR[-1])
			maxAct = self.outputR[-1][:,neuron]
			self.outputR[-1] = np.zeros(self.outputR[-1].shape,dtype='float32')
			self.outputR[-1][:,neuron] = maxAct
		for currLayer,k in zip(reversed(self.model.layers),range(self.numLayers-2,-1,-1)):
			nextLayer = currLayer._inbound_nodes[0].inbound_layers[0]
			activation = self.outputR[k-1] if (k-1!=-1) else imgData
			if self.optionRule == 'ZPlus':
				layerRules.append(ZPlus(currLayer,activation))
			elif self.optionRule == 'ZAlpha':
				layerRules.append(ZAlpha(currLayer,activation,self.alpha))
			else:
				assert False, 'This Rule Option is not supported'
			if verbose:
				print_msg('<><><><><>',option='verbose')
				print_msg('Weights From: ['+currLayer.name+']',option='verbose')
				print_msg('Activations From: ['+nextLayer.name+']',option='verbose')
		self.rules = layerRules
		print_msg('========== DONE ==========\n')

	# >> RUN_RULES: computes the relevance tensor for all the model layers.
	def run_rules(self,verbose=True):
		print_msg('Run Rules...')
		print_msg('--------------------------')
		R = {};
		R[self.rules[0].name] = K.identity(self.outputR[-1])
		if verbose:
			shape = print_tensor_shape(R[self.rules[0].name])
			print_msg('<><><><><>',option='verbose')
			print_msg('Rule R['+self.rules[0].name+'] Correctly runned.',option='verbose')
			print_msg('Tensor Output Shape: '+shape,option='verbose')
		for k in range(len(self.rules)):
			if k != len(self.rules)-1:
				R[self.rules[k+1].name] = self.rules[k].run(R[self.rules[k].name],ignoreBias=False)
				if verbose:
					shape = print_tensor_shape(R[self.rules[k+1].name])
					print_msg('<><><><><>',option='verbose')
					print_msg('Rule R['+self.rules[k+1].name+'] Correctly runned.',option='verbose')
					print_msg('Tensor Output Shape: '+shape,option='verbose')
			else:
				R['input'] = self.rules[k].run(R[self.rules[k].name],ignoreBias=False)
				if verbose:
					shape = print_tensor_shape(R['input'])
					print_msg('<><><><><>',option='verbose')
					print_msg('Rule R[input] Correctly runned.',option='verbose')
					print_msg('Tensor Output Shape: '+shape,option='verbose')
		print_msg('========== DONE ==========\n')

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath,cmap='plasma',option='sum'):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		oneDim = reduce_channels(self.relevance.copy(),option=option)
		heatMap = deprocess_image(oneDim.copy())
		visualize_heatmap(self.rawData,heatMap[0],self.__class__.__name__,cmap,savePath)
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of the LRP method
	def execute(self,fileName,sess,layerName=None,oneHot=False,verbose=False):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		self.define_rules(imgData,oneHot=oneHot,verbose=verbose)
		self.run_rules(verbose=verbose)
		if layerName is None:
			self.relevance = sess.run(self.R['input'])
		else:
			self.relevance = sess.run(self.R[layerName])
		print_msg('========== DONE ==========\n')
		return self.relevance

# -- Z+ RULE -- #
class ZPlus(object):
	def __init__(self,currLayer,nextAct):
		self.layer = currLayer
		self.name = currLayer.name
		self.type = self.layer.__class__.__name__
		self.act = K.identity(nextAct)
		self.maxValue = K.epsilon()
		self.minValue = -K.epsilon()
		
	def run(self,R,ignoreBias=True):
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
			
	def run_dense(self,R,ignoreBias=True):
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
		shape = self.act.get_shape().as_list()
		shape[0] = -1
		return K.reshape(R, shape)
	
	def run_pool(self,R):
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

# -- ZAlpha RULE -- #
class ZAlpha(object):
	def __init__(self,currLayer,nextAct,alpha):
		self.alpha = alpha
		self.beta = 1-alpha
		self.layer = currLayer
		self.name = currLayer.name
		self.type = self.layer.__class__.__name__
		self.act = K.identity(nextAct)
		self.maxValue = K.epsilon()
		self.minValue = -K.epsilon()
		
	def run(self,R,ignoreBias=True):
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
		
	def run_dense(self,R,ignoreBias=True):
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
		shape = self.act.get_shape().as_list()
		shape[0] = -1
		return K.reshape(R, shape)
	
	def run_pool(self,R):
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
