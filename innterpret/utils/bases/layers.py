from __future__ import absolute_import

# --  EXTERN IMPORT -- #
from keras.layers import Input, InputLayer, Dense, Conv2D, MaxPooling2D, Flatten, Activation
import tensorflow as tf
import keras.backend as K
import numpy as np
import math

# -- IMPORT -- #
from ..interfaces import DeconvLayer

class DConv2D(DeconvLayer):
	"""CLASS::DConv2D:
		---
		Description:
		---
		>Deconvolution Convolution 2D layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer
		# -- UP FUNCTION -- #
		weights = layer.get_weights()
		W,b = weights
		upRow,upCol,_,upFilters = W.shape
		upInput = Input(shape = layer.input_shape[1:])
		upOutput = Conv2D(upFilters,(upRow,upCol),kernel_initializer=tf.constant_initializer(W),
								   bias_initializer=tf.constant_initializer(b),padding='same')(upInput)
		self.up_function = K.function([upInput, K.learning_phase()],[upOutput])
		# -- DOWN FUNCTION -- #
		W = np.transpose(W,(0,1,3,2))
		W = W[::-1, ::-1,:,:]
		downRow,downCol,_,downFilters = W.shape
		b = np.zeros(downFilters)
		downInput = Input(shape = layer.output_shape[1:])
		downOutput = Conv2D(downFilters,(downRow,downCol),kernel_initializer=tf.constant_initializer(W),
								   bias_initializer=tf.constant_initializer(b),padding='same')(downInput)
		self.down_function = K.function([downInput, K.learning_phase()],[downOutput])

	def up(self, data, learn = 0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		self.downData = self.down_function([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

class DActivation(DeconvLayer):
	"""CLASS::DActivation:
		---
		Description:
		---
		>Deconvolution Activation layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer
		self.activation = layer.activation
		deconvInput = K.placeholder(shape = layer.output_shape)
		deconvOutput = self.activation(deconvInput)
		self.up_function = K.function([deconvInput, K.learning_phase()],[deconvOutput])
		self.down_function = K.function([deconvInput, K.learning_phase()],[deconvOutput])

	def up(self, data, learn = 0): 
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches.""" 
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		self.downData = self.down_function([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

class DInput(DeconvLayer):
	"""CLASS::DInput:
		---
		Description:
		---
		>Deconvolution Input layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer

	def up(self, data, learn = 0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		self.upData = data
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		data = np.expand_dims(data,axis=0)
		self.downData = data
		return self.downData

class DDense(DeconvLayer):
	"""CLASS::DDense: 
		---
		Description:
		---
		>Deconvolution Dense layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer
		weights = layer.get_weights()
		W, b = weights
		# -- UP FUNCTION -- #
		deconvInput = Input(shape = layer.input_shape[1:])
		deconvOutput = Dense(layer.output_shape[1],kernel_initializer=tf.constant_initializer(W),
							 bias_initializer=tf.constant_initializer(b))(deconvInput)
		self.up_function = K.function([deconvInput, K.learning_phase()], [deconvOutput])
		# -- DOWN FUNCTION -- #
		W = W.transpose()
		b = np.zeros(layer.input_shape[1])
		deconvInput = Input(shape = layer.output_shape[1:])
		deconvOutput = Dense(layer.input_shape[1:],kernel_initializer=tf.constant_initializer(W),
							 bias_initializer=tf.constant_initializer(b))(deconvInput)
		self.down_function = K.function([deconvInput, K.learning_phase()], [deconvOutput])

	def up(self, data, learn = 0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		self.downData = self.down_function([data, learn])
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

class DFlatten(DeconvLayer):
	"""CLASS::DFlatten:
		---
		Description:
		---
		>Deconvolution Flatten layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer
		self.shape = layer.input_shape[1:]
		self.up_function = K.function([layer.input, K.learning_phase()], [layer.output])

	def up(self,data,learn = 0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		self.upData = self.up_function([data, learn])
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		newShape = [data.shape[0]] + list(self.shape)
		#assert np.prod(self.shape) == np.prod(data.shape[1:])
		self.downData = np.reshape(data, newShape)

class DBatch(DeconvLayer):
	"""CLASS::DBatch: 
		---
		Description:
		---
		>Deconvolution Batch layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self,layer):
		self.layer = layer

	def up(self,data,learn=0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		self.mean = data.mean()
		self.std = data.std()
		self.upData = data
		self.upData -= self.mean
		self.upData /= self.std
		self.upData = np.squeeze(self.upData,axis=0)
		self.upData = np.expand_dims(self.upData,axis=0)
		return self.upData

	def down(self,data,learn=0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		self.downData = data
		self.downData += self.mean
		self.downData *= self.std
		self.downData = np.squeeze(self.downData,axis=0)
		self.downData = np.expand_dims(self.downData,axis=0)
		return self.downData

class DPooling(DeconvLayer):
	"""CLASS::DPooling:
		---
		Description:
		---
		>Deconvolution Pooling layer.
		Arguments:
		---
		>- layer: layer object that will be recreated for a deconv layer."""
	def __init__(self, layer):
		self.layer = layer
		self.poolsize = layer.pool_size

	def up(self, data, learn = 0):
		"""METHOD::UP: Forward Pass.
			---
			Arguments:
			---
			>- data {tensor} -- data recollected from the previous layer.
			>- learn {int} -- flag to determine if training is needed (default: {0})
			Returns: 
			---
			>- {tensor} -- The data collected, the switches."""
		[self.upData, self.switch] = self.__max_pooling_with_switch(data, self.poolsize)
		return self.upData

	def down(self, data, learn = 0):
		"""METHOD::DOWN: Backward Pass.
			---
			Arguments:
			---
			>- data: data recollected from the previous layer.
			>- learn: flag to determine if training is needed (default: {0})
			Returns:
			>- {tensor} -- The reconstructed data."""
		self.downData = self.__max_unpooling_with_switch(data, self.switch)
		return self.downData

	def __max_pooling_with_switch(self, data, poolsize):
		"""METHOD::__MAX_POOLING_WITH_SWITCH: Computes pooling with the recolected switches.
			---
			Arguments:
			---
			>- data {tensor} -- the data needed to pool.
			>- poolsize {tuple} -- the pool size specified.
			Returns:
			>- {list} -- A list containing pooled values and the switches."""
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

	def __max_unpooling_with_switch(self, data, switch):
		"""METHOD::__MAX_UNPOOLING_WITH_SWITCH: Reconstructs the sampling with the recolected switches.
			---
			Arguments:
			---
			>- data {tensor} -- data needed to unpool.
			>- switch {tensor} -- switches captured.
			Returns:
			---
			>- {tensor} -- the unpooled information."""
		tile = np.ones((math.floor(switch.shape[1]/data.shape[1]),math.floor(switch.shape[2]/data.shape[2])))
		tile = np.expand_dims(tile,axis=3)
		data = np.squeeze(data,axis=0)
		out = np.kron(data, tile)
		unpooled = out * switch
		unpooled = np.expand_dims(unpooled,axis=0)
		return unpooled