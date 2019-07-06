from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
import numpy as np
import keras.backend as K

class Gradient():
	""">> CLASS:GRADIENT: Computes the method of gradients for the model."""
	def __init__(self,model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.layerName = layerName
		#assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		inputData = self.model.inputs[0]
		outputLayer = self.model.get_layer(self.layerName)
		loss = K.mean(outputLayer.output)
		self.gradient = K.function([inputData], [K.gradients(loss, inputData)[0]])
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		gradImg = self.gradient([imgData])
		heatMap = np.sum(gradImg[0],axis=-1)
		heatMap[heatMap < np.mean(heatMap)] = 0
		self.heatMap = heatMap
		vrb.print_msg('========== DONE ==========\n')
		return self.heatMap

	def visualize(self,savePath,cmap='bone'):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		heatMap = deprocess_image(self.heatMap.copy())
		visualize_heatmap(self.rawData,heatMap,self.__class__.__name__,cmap,savePath)
		vrb.print_msg('========== DONE ==========\n')