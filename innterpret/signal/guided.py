from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
import PIL.Image as pilImage
import numpy as np
import tensorflow as tf
import keras.backend as K

class GuidedBackProp():
	""">> CLASS:GUIDEDBACKPROP: http://arxiv.org/abs/1412.6806."""
	def __init__(self,vgg16Model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		#assert self.vgg16Model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		if 'GuidedBackProp' not in ops._gradient_registry._registry:
			@ops.RegisterGradient("GuidedBackProp")
			def _GuidedBackProp(op, grad):
				dtype = op.inputs[0].dtype
				return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu':'GuidedBackProp'}):
			layerList = [layer for layer in vgg16Model.layers[1:] if hasattr(layer, 'activation')]
			for layer in layerList:
				if layer.activation.__name__ == 'relu':
					layer.activation = tf.nn.relu
			guidedModel = VGG16(weights='imagenet',include_top=True)
			modelInput = guidedModel.input
			layerOutput = guidedModel.get_layer(layerName).output
			argument = K.gradients(layerOutput,modelInput)[0]  
			self.gradient = K.function([modelInput],[argument])
			self.model = guidedModel
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		saliency = self.gradient([imgInput])
		self.saliency = saliency[0][0,:,:,:]
		vrb.print_msg('========== DONE ==========\n')
		return self.saliency

	def visualize(self,savePath):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize'+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = deprocess_image(self.saliency.copy())
		visualize_heatmap(self.rawData,result,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')