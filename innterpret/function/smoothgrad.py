from __future__ import absolute_import

# -- EXTERN IMPORT -- #
import numpy as np
import keras.backend as K

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from .gradient import Gradient

# -- SMOOTHGRAD METHOD -- #
class SmoothGrad(Gradient):
	"""CLASS::SmoothGrad:
		---
		Description:
		---
		> Method that reduces the noise from Gradient results:
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- The selected layer to analyze.
		Link:
		---
		>- http://arxiv.org/abs/1706.03825."""
	def __init__(self,model,layerName):
		super().__init__(model, layerName)

	def interpret(self,fileName,samples=50,stdNoise=10):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path to the image file.
			>- samples {int} -- Number of times Gradient will be applied. (default:{50})
			>- stdNoise {float} -- The standard deviation of the noise added. (default:{10})
			Returns:
			---
			>- {np.array} -- The saliency map."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData,imgData = load_image(fileName,preprocess=True)
		SmoothGrad = []
		for _ in range(samples):
			noiseSignal = np.random.normal(0,stdNoise,imgData.shape)
			img = imgData+noiseSignal
			gradVal = self.gradient([img])[0]
			SmoothGrad.append(gradVal)
		self.heatMap = np.mean(np.array(SmoothGrad),axis=0)
		self.heatMap = np.sum(self.heatMap[0],axis=-1)
		self.heatMap[self.heatMap < np.mean(self.heatMap)] = 0
		vrb.print_msg('========== DONE ==========\n')
		return self.heatMap
		
	def visualize(self,savePath,cmap='bone'):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- The path where the graph will be saved.
			>- cmap {string} -- The color map used to represent the graph.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		heatMap = deprocess_image(self.heatMap.copy())
		visualize_heatmap(self.rawData,heatMap,self.__class__.__name__,cmap,savePath)
		vrb.print_msg('========== DONE ==========\n')