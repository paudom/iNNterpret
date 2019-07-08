from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image,visualize_heatmap
from ..utils.bases.layers import DConv2D,DInput,DFlatten,DActivation,DPooling,DDense,DBatch
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
import PIL.Image as pilImage
import numpy as np
import tensorflow as tf
import keras.backend as K

class Deconvolution():
	""">> CLASS:DECONVOLUTION: http://arxiv.org/abs/1311.2901."""
	def __init__(self,model,layerName):
		self.numFilters = model.get_layer(layerName).output_shape[3]
		self.deconvLayers = []
		self.layerName = layerName
		for i in range(len(model.layers)):
			if model.layers[i].__class__.__name__ == 'Conv2D':
				self.deconvLayers.append(DConv2D(model.layers[i]))
				self.deconvLayers.append(DActivation(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'MaxPooling2D':
				self.deconvLayers.append(DPooling(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'Dense':
				self.deconvLayers.append(DDense(model.layers[i]))
				self.deconvLayers.append(DActivation(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'Activation':
				self.deconvLayers.append(DActivation(model.alyers[i]))
			elif model.layers[i].__class__.__name__ == 'Flatten':
				self.deconvLayers.append(DFlatten(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'InputLayer':
				self.deconvLayers.append(DInput(model.layers[i]))
			#else:
				#assert False, print_msg('The specified layer can not be handled for Deconvolution',show=False,option='error')
			if self.layerName == model.layers[i].name:
				break
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName,featVis,visMode='all'):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		assert 0 <= featVis <= self.numFilters
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		vrb.print_msg('Forward Pass')
		vrb.print_msg('--------------------------')
		self.deconvLayers[0].up(imgInput)
		for k in range(1, len(self.deconvLayers)):
			self.deconvLayers[k].up(self.deconvLayers[k - 1].upData)
		output = self.deconvLayers[-1].upData
		#assert output.ndim == 2 or output.ndim == 4
		if output.ndim == 2:
			featureMap = output[:,featVis]
		else:
			featureMap = output[:,:,:,featVis]
		if visMode == 'max':
			maxAct = featureMap.max()
			temp = featureMap == maxAct
			featureMap = featureMap * temp
		#elif visMode != 'all':
			#assert False, print_msg('This visualization mode is not implemented',show=False,option='error')
		output = np.zeros_like(output)
		if 2 == output.ndim:
			output[:,featVis] = featureMap
		else:
			output[:,:,:,featVis] = featureMap
		vrb.print_msg('Backward Pass')
		vrb.print_msg('--------------------------')
		self.deconvLayers[-1].down(output)
		for k in range(len(self.deconvLayers)-2,-1,-1):
			self.deconvLayers[k].down(self.deconvLayers[k + 1].downData)
		deconv = self.deconvLayers[0].downData
		self.deconv = deconv.squeeze()
		vrb.print_msg('========== DONE ==========\n')
		return self.deconv

	def visualize(self,savePath):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize'+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = (self.deconv-self.deconv.min())/(self.deconv.max()-self.deconv.min()+K.epsilon())
		uint8Deconv = (result* 255).astype(np.uint8)
		img = pilImage.fromarray(uint8Deconv, 'RGB')
		visualize_heatmap(self.rawData,img,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')
