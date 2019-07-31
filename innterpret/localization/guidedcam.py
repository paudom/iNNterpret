from __future__ import absolute_import

# -- EXTERN IMPORT -- #
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
from PIL import Image as pilImage
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.tensor import load_model, load_vgg16, load_vgg19
from ..utils.tensor import decode_predictions
from ..utils.interfaces import Method
from ..utils.exceptions import NotAValidTensorError

class GuidedGradCAM(Method):
	"""CLASS::GuidedGradCAM:
		---
		Description:
		---
		> Show the pixels which the model is using to decide a certain class.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- The selected layer to analyze.
		>- option {string} -- Model to load. (default:{'vgg16'})
		>>- 'vgg16'
		>>- 'vgg19'
		>>- 'other'.
		>- h5file {strin} -- Path to the h5file, specify if option is 'other'.(default:{None}).
		Raises:
		---
		>- NotAValidTensorError {Exception} -- If the layer specified is not a convolution layer.
		Link:
		---
		>- http://arxiv.org/abs/1610.02391."""
	def __init__(self,model,layerName,option='vgg16',h5file=None):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		if self.model.get_layer(layerName).__class__.__name__ != 'Conv2D':
			raise NotAValidTensorError('The layer "'+layerName+'" is not a convolution layer.')
		self.layerName = layerName
		self.numClasses = self.model.outputs[0].get_shape()[1]-1
		if 'GuidedBackProp' not in ops._gradient_registry._registry:
			@ops.RegisterGradient("GuidedBackProp")
			def _GuidedBackProp(op, grad):
				dtype = op.inputs[0].dtype
				return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu':'GuidedBackProp'}):
			layerList = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
			for layer in layerList:
				if layer.activation.__name__ == 'relu':
					layer.activation = tf.nn.relu
			if option == 'vgg19':
				self.guidedModel = load_vgg19()
			elif option == 'vgg16':
				self.guidedModel = load_vgg16()
			else:
				self.guidedModel = load_model(h5file)
			modelInput = self.guidedModel.input
			layerOutput = self.guidedModel.get_layer(layerName).output
			argument = K.gradients(layerOutput,modelInput)[0]  
			self.gradient = K.function([modelInput],[argument])
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName,topCls=5,negGrad=False):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path of the image data.
			>- topCls {int} -- The number classes with the highest propability to show. (default:{5}).
			>- negGrad {bool} -- Flag to determine how the gradients are computed. (default:{False}).
			Returns:
			---
			>- {np.array} -- A heat map representing the areas where the model is focussing.
			Raises:
			---
			>- ValueError {Exception} -- If the selected class is not valid."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData,imgInput = load_image(fileName,preprocess=True)
		decoded = decode_predictions(self.model.predict(imgInput),top=topCls)
		vrb.print_msg('Predicted classes: '+str(decoded))
		selClass = int(input(vrb.set_msg('Select the class to explain (0-'+str(self.numClasses)+'): ')))
		if not 0 <= selClass <= self.numClasses:
			raise ValueError('The class selected is invalid. It has to be between [0,'+
								self.numClasses+'].')
		clScore = self.model.output[0, selClass]
		convOutput = self.model.get_layer(self.layerName).output
		grads = K.gradients(clScore, convOutput)[0]
		if negGrad:
			vrb.print_msg('Setting Negative Explanation.')
			grads = -grads
		vrb.print_msg('Computing HeatMap')
		vrb.print_msg('--------------------------')
		self.camGrad = K.function([self.model.input],[convOutput, grads])
		output, gradsVal = self.camGrad([imgInput])
		weights = np.mean(gradsVal[0, :, :, :],axis=(0,1))
		self.cam = np.dot(output[0, :],weights)
		self.cam = np.asarray(pilImage.fromarray(self.cam).resize((224,224),pilImage.BILINEAR),dtype='float32')
		self.cam = np.maximum(self.cam, K.epsilon())
		self.cam = self.cam/self.cam.max()
		vrb.print_msg('Computing Guided')
		vrb.print_msg('--------------------------')
		saliency = self.gradient([imgInput])
		threshCAM = np.mean(self.cam)*0.8
		gcam = cam.copy()
		gcam[gcam<threshCAM] = 0.0
		self.guidedCAM = saliency[0][0,:,:,:]*gcam[...,np.newaxis]
		vrb.print_msg('========== DONE ==========\n')
		return self.guidedCAM

	def visualize(self,savePath):
		"""METHOD::VISUALIZE
			---
			Arguments:
			---
			>- savePath {string} -- The path where the graph will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = deprocess_image(self.guidedCAM.copy())
		visualize_heatmap(self.rawData,result,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')