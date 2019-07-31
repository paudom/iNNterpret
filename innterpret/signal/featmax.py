from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import deprocess_image
from ..utils.interfaces import Method
from ..utils.exceptions import NotAValidTensorError
from keras.preprocessing import image as kerasImage
from PIL import Image as pilImage
import keras.backend as K
import numpy as np
import imageio
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

class FeatMaximization(Method):
	"""CLASS::FeatMaximization:
		---
		Description:
		---
		> Returns an image representing the pattern learned from the model.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- The name of the layer to analyze, it has to be a convolution layer.
		>- featureMap {int} -- The feature map to visualize.
		>- targetSize {int} -- The size of the resulting image (height = width).
		Link:
		---
		>- http://arxiv.org/abs/1312.6034."""
	def __init__(self,model,layerName,featureMap,targetSize):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		self.factor = 1.2; self.upSteps = 9
		self.model = model
		self.input = self.model.inputs[0]
		if self.model.get_layer(layerName).__class__.__name__ != 'Conv2D':
			raise NotAValidTensorError('The layer "'+layerName+'" is not a convolution layer')
		self.layerName = layerName
		self.layer = self.model.get_layer(self.layerName)
		if not 0 <= featureMap <= self.layer.shape[-1]-1:
			raise ValueError('The feature Map needs to be between the interval [0,'+
								self.layer.shape[-1]-1+'].')
		loss = K.mean(self.layer[:,:,:,featureMap])
		grads = K.gradients(loss,self.input)[0]
		grads /= K.sqrt(K.mean(K.square(grads)))+K.epsilon()
		self.gradient = K.function([self.input],[loss,grads])
		self.targetSize = targetSize
		self.size = int(self.targetSize/(self.factor**self.upSteps))
		self.imgData = np.random.normal(0,10,(1,self.size,self.size,3))
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,epochs):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- epochs {int} -- Number of iterations.
			Returns:
			---
			{np.array} -- Image representing the pattern learned from the model."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.gifImg = []
		self.gifImg.append(self.imgData[0].copy())
		for up in reversed(range(self.upSteps)):
			for _ in range(epochs):
				lossVal,gradVal = self.gradient([self.imgData])
				if lossVal <= K.epsilon():
					vrb.print_msg('Gradient got stuck')
				break
			step = 1/(gradVal.std()+K.epsilon())
			self.imgData += gradVal*step
			self.gifImg.append(self.imgData[0].copy())
			vrb.print_msg('Current loss value: '+str(lossVal))
			size = int(self.targetSize/(self.factor**up))
			img = deprocess_image(self.imgData[0],scale=0.25)
			img = np.asarray(pilImage.fromarray(img).resize((size,size),pilImage.BILINEAR),dtype='float32')
			self.imgData = [self.__process_image(img,self.imgData[0])]
		img = deprocess_image(self.imgData[0])
		self.actMax = img
		vrb.print_msg('========== DONE ==========\n')
		return self.actMax

	def __process_image(self,x,previous):
		"""METHOD::__PROCESS_IMAGE:
			---
			Arguments:
			---
			>- x {np.array} -- Array representing the image to process.
			>- previous {np.array} -- Array representing the previous image.
			Returns:
			---
			>- {np.array} -- Array representing a valid image."""
		x = x/255; x -= 0.5
		return x*4*previous.std()+previous.mean()

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
		plt.imshow(self.actMax)
		plt.show()
		img = kerasImage.array_to_img(self.actMax,scale=False)
		img.save(savePath,dpi=250)
		vrb.print_msg('========== DONE ==========\n')

	def produce_gif(self,savePath):
		"""METHOD::PRODUCE_GIF: generate a gif with all the iterations.
			---
			Arguments:
			---
			>- savePath {string} -- The path where the gif will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Produce '+self.__class__.__name__+' gif...')
		vrb.print_msg('--------------------------')
		size = int(self.targetSize/2) 
		with imageio.get_writer(savePath, mode='I') as writer:
			for im in self.gifImg:
				image = deprocess_image(im.copy())
				image = np.asarray(pilImage.fromarray(image).resize((size,size),pilImage.ANTIALIAS))
				writer.append_data(image)
		vrb.print_msg('========== DONE ==========\n')
	
	def produce_mosaic(self,samples,savePath):
		"""METHOD::PRODUCE_MOSAIC: produce a mosaic with the images generated on some iterations.
			---
			Arguments:
			---
			>- samples {int} -- The number of samples to skip.
			>- savePath {string} -- The path where the mosaic will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Produce '+self.__class__.__name__+' mosaic...')
		vrb.print_msg('--------------------------')
		margin = 5
		size = int(self.targetSize/2) 
		mosaic = []
		for im in self.gifImg[0::samples]:
			image = deprocess_image(im.copy())
			mosaic.append(np.asarray(pilImage.fromarray(image).resize((size,size),pilImage.ANTIALIAS)))
		n = int(np.round(np.sqrt(len(mosaic))))
		cols = size*n+(n-1)*margin
		rows = size*n+(n-1)*margin
		draw = np.zeros((cols,rows,3),dtype='uint8')
		stop = False; im = 0
		for c in range(n):
			if not stop:
				for r in range(n):
					wM = (size+margin)*c
					hM = (size+margin)*r
					draw[wM:wM+size,hM:hM+size,:] = mosaic[im]
					im += 1
					if(im >= len(mosaic)):
						stop = True
						break
			else:
				break
		imgDraw = kerasImage.array_to_img(draw,scale=False)
		imgDraw.save(savePath,dpi=(250,250))
		vrb.print_msg('========== DONE ==========\n')