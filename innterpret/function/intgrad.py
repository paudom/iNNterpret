from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, visualize_heatmap
from .gradient import Gradient
import numpy as np
from keras.preprocessing import image as kerasImage
from PIL import Image, ImageEnhance

class IntegratedGrad(Gradient):
	"""CLASS::IntegratedGrad:
		---
		Description:
		---
		> Method that reduces the noise of Gradient results:
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- Layer selected to Analyze.
		Link:
		---
		>- http://arxiv.org/abs/1703.01365."""
	def __init__(self,model,layerName):
		super().__init__(model, layerName)

	def interpret(self,fileName,samples=50):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path to the image file.
			>- samples {int} -- the number of samples between baseline and limit. (default:{50}).
			Returns:
			---
			>- {np.array} -- Masked image indicating the pixels that change activations."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		vrb.print_msg('Computing Baseline and Limit')
		vrb.print_msg('--------------------------') 
		self.__luminance_interpolation(self.rawData, 0.0, 1.0, 50)
		imgData = kerasImage.img_to_array(self.rawData)
		imgData = np.expand_dims(imgData,axis=0)
		self.imgData = imgData
		fullPred = self.model.predict(self.imgData)
		maxClass = np.argsort(fullPred)[0][-1:][::-1]
		predictions = []
		for k in range(50):
			imgInput = kerasImage.img_to_array(self.imgArray[k])
			imgInput = np.expand_dims(imgInput,axis=0)
			predictions.append(self.model.predict(imgInput)[0][maxClass][0])
		maxBase = np.max(np.array(predictions))*0.9
		minBase = np.max(np.array(predictions))*0.1
		baseline = [n for n,i in enumerate(predictions) if i>minBase][0]
		limit = [n for n,i in enumerate(predictions) if i>maxBase][0]
		vrb.print_msg('Baseline: '+str(self.interY[baseline]))
		vrb.print_msg('Limit: '+str(self.interY[limit]))
		self.__luminance_interpolation(self.rawData,self.interY[baseline],self.interY[limit],samples)
		vrb.print_msg('Computing Integrated Gradients')
		vrb.print_msg('--------------------------')
		IntGrad = []
		for k in range(samples):
			imgInput = kerasImage.img_to_array(self.imgArray[k])
			imgInput = np.expand_dims(imgInput,axis=0)
			gradVal = self.gradient([imgInput])[0]
			IntGrad.append(gradVal)
		heatMap = np.mean(np.array(IntGrad),axis=0)
		heatMap = np.sum(heatMap[0],axis=-1)
		heatMap[heatMap < np.mean(heatMap)] = 0
		self.heatMap = heatMap
		vrb.print_msg('========== DONE ==========\n')
		return self.heatMap

	def __luminance_interpolation(self,imgLum,a,b,samples):
		"""METHOD::__LUMINANCE_INTERPOLATION: returns an array with images with different brightness.
			---
			Arguments:
			---
			>- imgLum {np.array} -- image from which create different samples.
			>- a {float} -- Initial alpha value.
			>- b {float} -- Final alpha value.
			>- samples {int} -- The number of samples between 'a' and 'b'.
			Returns:
			---
			>- {list[np.array]} -- Containing all the images generated."""
		self.imgArray = []
		self.interY = np.linspace(a,b,num=samples)
		self.lum = ImageEnhance.Brightness(imgLum)
		for k in range(samples):
			self.imgArray.append(self.lum.enhance(self.interY[k]))

	def visualize(self,savePath):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- The path where the graph will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		mask = np.zeros(self.imgData.shape)
		IntGradNorm = (self.heatMap - np.min(self.heatMap))/(np.max(self.heatMap)-np.min(self.heatMap))
		mask[0,:,:,0] = IntGradNorm; mask[0,:,:,1] = IntGradNorm; mask[0,:,:,2] = IntGradNorm
		IntGradImg = self.imgData[0]*mask
		IntGradImg = kerasImage.array_to_img(IntGradImg[0])
		visualize_heatmap(self.rawData,IntGradImg,self.__class__.__name__,'bone',savePath)
		vrb.print_msg('========== DONE ==========\n')