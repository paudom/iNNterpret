from __future__ import absolute_import

# --  IMPORT -- #
from keras.layers import Input, InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Activation
import tensorflow as tf
import keras.backend as K
import numpy as np
import math

# -- DCONV2D LAYER -- #
class DConv2D(object):
	def __init__(self, layer):
		self.layer = layer
		# -- UP FUNCTION -- #
		weights = layer.get_weights()
		W = weights[0]
		b = weights[1]
		upFilters = W.shape[3]
		upRow = W.shape[0]
		upCol = W.shape[1]
		upInput = Input(shape = layer.input_shape[1:])
		upOutput = Conv2D(upFilters,(upRow,upCol),kernel_initializer=tf.constant_initializer(W),
								   bias_initializer=tf.constant_initializer(b),padding='same')(upInput)
		self.up_function = K.function([upInput, K.learning_phase()],[upOutput])
		# -- DOWN FUNCTION -- #
		W = np.transpose(W,(0,1,3,2))
		W = W[::-1, ::-1,:,:]
		downFilters = W.shape[3]
		downRow = W.shape[0]
		downCol = W.shape[1]
		b = np.zeros(downFilters)
		downInput = Input(shape = layer.output_shape[1:])
		downOutput = Conv2D(downFilters,(downRow,downCol),kernel_initializer=tf.constant_initializer(W),
								   bias_initializer=tf.constant_initializer(b),padding='same')(downInput)
		self.down_function = K.function([downInput, K.learning_phase()],[downOutput])
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		self.downData = self.down_function([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

# -- DACTIVATION LAYER -- #
class DActivation(object):
	def __init__(self, layer, linear = False):
		self.layer = layer
		self.linear = linear
		self.activation = layer.activation
		deconvInput = K.placeholder(shape = layer.output_shape)
		deconvOutput = self.activation(deconvInput)
		self.up_function = K.function([deconvInput, K.learning_phase()],[deconvOutput])
		self.down_function = K.function([deconvInput, K.learning_phase()],[deconvOutput])
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):   
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		self.downData = self.down_function([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

# -- DINPUT LAYER -- #
class DInput(object):
	def __init__(self, layer):
		self.layer = layer
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):
		self.upData = data
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		data = np.expand_dims(data,axis=0)
		self.downData = data
		return self.downData

# -- DDENSE LAYER -- #
class DDense(object):
	def __init__(self, layer):
		self.layer = layer
		weights = layer.get_weights()
		W = weights[0]
		b = weights[1]
		
		# -- UP FUNCTION -- #
		deconvInput = Input(shape = layer.input_shape[1:])
		deconvOutput = Dense(layer.output_shape[1],kernel_initializer=tf.constant_initializer(W),
							 bias_initializer=tf.constant_initializer(b))(deconvInput)
		self.up_function = K.function([deconvInput, K.learning_phase()], [deconvOutput])
		
		# -- DOWN FUNCTION -- #
		W = W.transpose()
		self.inputShape = layer.input_shape
		self.outputShape = layer.output_shape
		b = np.zeros(self.inputShape[1])
		deconvInput = Input(shape = self.outputShape[1:])
		deconvOutput = Dense(self.input_shape[1:],kernel_initializer=tf.constant_initializer(W),
							 bias_initializer=tf.constant_initializer(b))(deconvInput)
		self.down_function = K.function([deconvInput, K.learning_phase()], [deconvOutput])
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		self.downData = self.down_func([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

# -- DFLATTEN LAYER -- #
class DFlatten(object):
	def __init__(self, layer):
		self.layer = layer
		self.shape = layer.input_shape[1:]
		self.up_function = K.function([layer.input, K.learning_phase()], [layer.output])
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):
		self.upData = self.up_func([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		newShape = [data.shape[0]] + list(self.shape)
		assert np.prod(self.shape) == np.prod(data.shape[1:])
		self.downData = np.reshape(data, newShape)

# -- DBATCH LAYER -- #
class DBatch(object):
	def __init__(self,layer):
		self.layer = layer
	# -- FORWARD PASS -- #
	def up(self,data,learn=0):
		self.mean = data.mean()
		self.std = data.std()
		self.upData = data
		self.upData -= self.mean
		self.upData /= self.std
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self,data,learn=0):
		self.downData = data
		self.downData += self.mean
		self.downData *= self.std
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData
		return self.downData

# -- DPOOLING LAYER -- #
class DPooling(object):
	def __init__(self, layer):
		self.layer = layer
		self.poolsize = layer.pool_size
	# -- FORWARD PASS -- #
	def up(self, data, learn = 0):
		[self.upData, self.switch] = self.__max_pooling_with_switch(data, self.poolsize)
		return self.upData
	# -- BACKWARD PASS -- #
	def down(self, data, learn = 0):
		self.downData = self.__max_unpooling_with_switch(data, self.switch)
		return self.downData
	# -- POOLING -- #
	def __max_pooling_with_switch(self, data, poolsize):
		switch = np.zeros(data.shape)
		outShape = list(data.shape)
		rowPool = int(poolsize[0])
		colPool = int(poolsize[1])
		outShape[1] = math.floor(outShape[1] / poolsize[0])
		outShape[2] = math.floor(outShape[2] / poolsize[1])
		pooled = np.zeros(outShape)
		for sample in range(data.shape[0]):
			for dim in range(data.shape[3]):
				for row in range(outShape[1]):
					for col in range(outShape[2]):
						patch = data[sample, 
								row * rowPool : (row + 1) * rowPool,
								col * colPool : (col + 1) * colPool,
								dim]
						maxVal = patch.max()
						pooled[sample,row,col,dim] = maxVal
						maxIndex = patch.argmax(axis = 1)
						maxRow = patch.max(axis = 1).argmax()
						maxCol = maxIndex[maxRow]
						switch[sample, 
							   row * rowPool + maxRow, 
							   col * colPool + maxCol,
							   dim]  = 1
		return [pooled, switch]
	# -- UNPOOLING -- #
	def __max_unpooling_with_switch(self, data, switch):
		tile = np.ones((math.floor(switch.shape[1]/data.shape[1]),math.floor(switch.shape[2]/data.shape[2])))
		tile = np.expand_dims(tile,axis=3)
		data = np.squeeze(data,axis=0)
		out = np.kron(data, tile)
		unpooled = out * switch
		unpooled = np.expand_dims(unpooled,axis=0)
		return unpooled