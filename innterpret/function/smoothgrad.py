from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from .gradient import Gradient
import numpy as np
import keras.backend as K

# -- SMOOTHGRAD METHOD -- #
class SmoothGrad(Gradient):
	""">> CLASS:SMOOTHGRAD: Method that reduces the noise from Gradient results:
		http://arxiv.org/abs/1706.03825."""
	def __init__(self,model,layerName):
		super().__init__(model, layerName)

	def execute(self,fileName,samples=50,stdNoise=10):
		""">> EXECUTE: returns the result of the method"""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		SmoothGrad = []
		for _ in range(samples):
			noiseSignal = np.random.normal(0,stdNoise,imgData.shape)
			img = imgData+noiseSignal
			gradVal = self.gradient([img])[0]
			SmoothGrad.append(gradVal)
		heatMap = np.mean(np.array(SmoothGrad),axis=0)
		heatMap = np.sum(heatMap[0],axis=-1)
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