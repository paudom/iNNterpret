from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.data import load_image, deprocess_image visualize_heatmap
import numpy as np
import keras.backend as K
from keras.preprocessing import image as kerasImage
from PIL import Image, ImageEnhance

# -- GRADIENT METHOD -- #
class Gradient():
	def __init__(self,model,layerName):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------')
		assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		try:
			inputData = model.inputs[0]
			outputLayer = model.get_layer(layerName)
			loss = K.mean(outputLayer.output)
			gradients = K.gradients(loss, inputData)[0]
			self.gradient = K.function([inputData], [gradients])
			self.model = model
			print_msg('========== DONE ==========\n')
		except ValueError as e:
			assert False, print_msg('The specified layer does not exist in the model introduced',show=False,option='error')

	# >> EXECUTE: returns the result of the GRADIENT method
	def execute(self,fileName):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		gradImg = self.gradient([imgData])
		heatMap = np.sum(gradImg[0],axis=-1)
		heatMap[heatMap < np.mean(heatMap)] = 0
		self.heatMap = heatMap
		print_msg('========== DONE ==========\n')
		return self.heatMap

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath,cmap='bone'):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		heatMap = deprocess_image(self.heatMap.copy())
		visualize_heatmap(self.rawData,heatMap,self.__class__.__name__,cmap,savePath)
		print_msg('========== DONE ==========\n')

# -- SMOOTHGRAD METHOD -- #
class SmoothGrad(Gradient):
	def __init__(self,model,layerName):
		super().__init__(model, layerName)

	# >> EXECUTE: returns the result of the GRADIENT method
	def execute(self,fileName,samples=50,stdNoise=10):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		smoothImg = self.gradient([imgData])
		SmoothGrad = []
		for k in range(samples):
			noiseSignal = np.random.normal(0,stdNoise,imgData.shape)
			img = imgData+noiseSignal
			gradVal = self.gradient([img])[0]
			SmoothGrad.append(gradVal)
		heatMap = np.mean(np.array(SmoothGrad),axis=0)
		heatMap = np.sum(heatMap[0],axis=-1)
		heatMap[heatMap < np.mean(heatMap)] = 0
		self.heatMap = heatMap
		print_msg('========== DONE ==========\n')
		return self.heatMap

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath,cmap='bone'):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		heatMap = deprocess_image(self.heatMap.copy())
		visualize_heatmap(self.rawData,heatMap,self.__class__.__name__,cmap,savePath)
		print_msg('========== DONE ==========\n')

class IntegratedGrad(Gradient):
	def __init__(self,model,layerName):
		super().__init__(model, layerName)

	# >> EXECUTE: returns the result of the GRADIENT method
	def execute(self,fileName,samples=50,verbose=True):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		print_msg('Computing Baseline and Limit')
		print_msg('--------------------------') 
		self.luminance_interpolation(self.rawData, 0.0, 1.0, 50)
		imgData = kerasImage.img_to_array(self.rawData)
		imgData = np.expand_dims(imgData,axis=0)
		self.imgData = imgData
		fullPred = self.model.predict(self.imgData)
		maxClass = np.argsort(fullPred)[0][-1:][::-1]
		predictions = []
		for k in range(50):
			imgInput = kerasImage.img_to_array(self.imgArray[k])
			imgInput = np.expand_dims(imgInput,axis=0)
			predictions.append(model.predict(imgInput)[0][maxClass][0])
		maxBase = np.max(np.array(predictions))*0.9
		minBase = np.max(np.array(predictions))*0.1
		baseline = [n for n,i in enumerate(predictions) if i>minBase][0]
		limit = [n for n,i in enumerate(predictions) if i>maxBase][0]
		if verbose:
			print_msg('Baseline: '+str(self.interY[baseline]),option='verbose')
			print_msg('Limit: '+str(self.interY[limit]))
		self.luminance_interpolation(self.rawData,self.interY[baseline],self.interY[limit],samples)
		print_msg('Computing Integrated Gradients')
		print_msg('--------------------------')
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
		print_msg('========== DONE ==========\n')
		return self.heatMap

	# >> LUMINANCE_INTERPOLATION: returns an array with images with different brightness
	def luminance_interpolation(self,imgLum,a,b,samples):
		self.imgArray = []
		self.interY = np.linspace(a,b,num=samples)
		self.lum = ImageEnhance.Brightness(imgLum)
		for k in range(samples):
			self.imgArray.append(self.lum.enhance(self.interY[k]))

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath,cmap='bone'):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		mask = np.zeros(self.imgData.shape)
		IntGradNorm = (self.heatMap - np.min(self.heatMap))/(np.max(self.heatMap)-np.min(self.heatMap))
		mask[0,:,:,0] = IntGradNorm; mask[0,:,:,1] = IntGradNorm; mask[0,:,:,2] = IntGradNorm
		IntGradImg = self.imgData[0]*mask
		IntGradImg = kerasImage.array_to_img(IntGradImg[0])
		visualize_heatmap(self.rawData,IntGradImg,self.__class__.__name__,cmap,savePath)
		print_msg('========== DONE ==========\n')