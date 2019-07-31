from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.interfaces import Method
from ..utils.exceptions import NotAValidTensorError
import numpy as np
import keras.backend as K

class Gradient(Method):
	"""CLASS::Gradient:
		---
		Description:
		---
		> Computes the method of gradients for the model.
		Arguments:
		---
		>- model {keras.Models} -- Model to analyze.
		>- layerName {string} -- The selected layer to visualize."""
	def __init__(self,model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		if self.model.get_layer(layerName).__class__.__name__ != 'Conv2D':
			raise NotAValidTensorError('The layer "'+layerName+'" is not a convolutional layer.')
		self.layerName = layerName
		inputLayer = self.model.inputs[0]
		outputLayer = self.model.get_layer(self.layerName)
		loss = K.mean(outputLayer.output)
		self.gradient = K.function([inputLayer], [K.gradients(loss, inputLayer)[0]])
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path to the image file.
			Returns:
			---
			>- {np.array} -- The saliency of the image."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData,imgData = load_image(fileName,preprocess=True)
		gradImg = self.gradient([imgData])
		self.heatMap = np.sum(gradImg[0],axis=-1)
		self.heatMap[self.heatMap < np.mean(heatMap)] = 0
		vrb.print_msg('========== DONE ==========\n')
		return self.heatMap

	def visualize(self,savePath,cmap='bone'):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- Path where the graph will be saved.
			>- cmap {string} -- color map used to see the saliency map.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		heatMap = deprocess_image(self.heatMap.copy())
		visualize_heatmap(self.rawData,heatMap,self.__class__.__name__,cmap,savePath)
		vrb.print_msg('========== DONE ==========\n')