from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.tensor import decode_predictions
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
from PIL import Image as pilImage
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

class GuidedGradCAM():
	""">> CLASS:GRADCAM: http://arxiv.org/abs/1610.02391."""
	def __init__(self,vgg16Model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = vgg16Model
		self.layerName = layerName
		#assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		self.numClasses = self.model.outputs[0].get_shape()[1]-1
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
			self.guidedModel = guidedModel
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName,topCls=5,negGrad=False):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		decoded = decode_predictions(self.model.predict(imgInput),top=topCls)
		vrb.print_msg('Predicted classes: '+str(decoded))
		clSel = int(input(vrb.set_msg('Select the class to explain (0-'+str(self.numClasses)+'): ')))
		clScore = self.model.output[0, clSel]
		convOutput = self.model.get_layer(self.layerName).output
		grads = K.gradients(clScore, convOutput)[0]
		if negGrad:
			vrb.print_msg('Negative Explanation')
			grads = -grads
		vrb.print_msg('Computing HeatMap')
		vrb.print_msg('--------------------------')
		self.camGrad = K.function([self.model.input],[convOutput, grads])
		output, gradsVal = self.camGrad([imgInput])
		output, gradsVal = output[0, :], gradsVal[0, :, :, :]
		weights = np.mean(gradsVal,axis=(0,1))
		cam = np.dot(output,weights)
		cam = np.asarray(pilImage.fromarray(cam).resize((224,224),pilImage.BILINEAR),dtype='float32')
		cam = np.maximum(cam, K.epsilon())
		self.cam = cam/cam.max()
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
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = deprocess_image(self.guidedCAM.copy())
		visualize_heatmap(self.rawData,result,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')