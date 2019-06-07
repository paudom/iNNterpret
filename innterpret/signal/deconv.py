from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.data import load_image,visualize_heatmap
from .layers import DConv2D,DInput,DFlatten,DActivation,DPooling,DDense,DBatch
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
import PIL.Image as pilImage
import tensorflow as tf
import keras.backend as K


# -- DECONVOLUTION METHOD -- #
class Deconvolution():
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
			else:
				assert False, print_msg('The specified layer can not be handled for Deconvolution',show=False,option='error')
			if self.layerName == model.layers[i].name:
				break
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of Deconvolution method
	def execute(self,fileName,featVis,visMode='all'):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		assert 0 <= featVis <= self.numFilters
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		print_msg('Forward Pass')
		print_msg('--------------------------')
		self.deconvLayers[0].up(data)
		for k in range(1, len(self.deconvLayers)):
			self.deconvLayers[k].up(self.deconvLayers[k - 1].upData)
		output = self.deconvLayers[-1].upData
		assert output.ndim == 2 or output.ndim == 4
		if output.ndim == 2:
			featureMap = output[:,featVis]
		else:
			featureMap = output[:,:,:,featVis]
		if visMode == 'max':
			maxAct = featureMap.max()
			temp = featureMap == maxAct
			featureMap = featureMap * temp
		elif visMode != 'all':
			assert False, print_msg('This visualization mode is not implemented',show=False,option='error')
		output = np.zeros_like(output)
		if 2 == output.ndim:
			output[:,featVis] = featureMap
		else:
			output[:,:,:,featVis] = featureMap
		print_msg('Backward Pass')
		print_msg('--------------------------')
		self.deconvLayers[-1].down(output)
		for k in range(len(self.deconvLayers)-2,-1,-1):
			self.deconvLayers[k].down(self.deconvLayers[k + 1].downData)
		deconv = self.deconvLayers[0].downData
		self.deconv = deconv.squeeze()
		print_msg('========== DONE ==========\n')
		return self.deconv

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath):
		print_msg('Visualize'+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		result = (self.deconv-self.deconv.min())/(self.deconv.max()-self.deconv.min()+K.epsilon())
		uint8Deconv = (result* 255).astype(np.uint8)
		img = pilImage.fromarray(uint8Deconv, 'RGB')
		visualize_heatmap(self.rawData,img,self.__class__.__name__,'viridis',savePath)
		print_msg('========== DONE ==========\n')

# -- GUIDED BACKPROPAGATION -- # 
class GuidedBackProp():
	def __init__(self,vgg16Model,layerName):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------')
		assert self.vgg16Model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		if 'GuidedBackProp' not in ops._gradient_registry._registry:
			@ops.RegisterGradient("GuidedBackProp")
			def _GuidedBackProp(op, grad):
				dtype = op.inputs[0].dtype
				return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu':'GuidedBackProp'}):
			layerList = [layer for layer in vgg16Model.layers[1:] if hasattr(layer, 'activation')]
			for layer in layerList:
				if layer.activation == relu:
					layer.activation = tf.nn.relu
			guidedModel = VGG16(weights='imagenet',include_top=True)
			modelInput = guidedModel.input
			layerOutput = guidedModel.get_layer(layerName).output
			argument = K.gradients(layerOutput,modelInput)[0]  
			self.gradient = K.function([modelInput],[argument])
			self.model = guidedModel
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of Guided Backpropagation method
	def execute(self,fileName):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		saliency = self.gradient([imgInput])
		self.saliency = saliency[0][0,:,:,:]
		print_msg('========== DONE ==========\n')
		return self.saliency

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath):
		print_msg('Visualize'+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		result = deprocess_image(self.saliency.copy())
		visualize_heatmap(self.rawData,result,self.__class__.__name__,'viridis',savePath)
		print_msg('========== DONE ==========\n')
