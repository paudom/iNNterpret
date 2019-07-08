from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import deprocess_image
from ..utils.tensor import model_remove_softmax
from keras.preprocessing import image as kerasImage
from scipy.ndimage.filters import gaussian_filter, median_filter
from PIL import Image as pilImage
import keras.backend as K
import numpy as np
import tensorflow as tf
import imageio
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

class FeatMaximization():
	""">> CLASS:FEATMAXIMIZATION: http://arxiv.org/abs/1312.6034."""
	def __init__(self,model,layerName,filt,targetSize):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		self.factor = 1.2; self.upSteps = 9
		self.model = model
		self.imgInput = self.model.inputs[0]
		self.layerName = layerName
		#assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		self.layer = self.model.get_layer(self.layerName)
		#numFilters = self.layer.shape[-1]
		#assert 0 <= filt <= numFilters-1
		loss = K.mean(self.layer[:,:,:,filt])
		grads = K.gradients(loss,self.imgInput)[0]
		grads /= K.sqrt(K.mean(K.square(grads)))+K.epsilon()
		self.gradient = K.function([self.imgInput],[loss,grads])
		self.targetSize = targetSize
		self.size = int(self.targetSize/(self.factor**self.upSteps))
		self.imgData = np.random.normal(0,10,(1,self.size,self.size,3))
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,epochs,verbose=True):
		""">> EXECUTE: returns the result of the method."""
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
			self.imgData = [self.process_image(img,self.imgData[0])]
		img = deprocess_image(self.imgData[0])
		self.actMax = img
		vrb.print_msg('========== DONE ==========\n')
		return self.actMax

	def process_image(self,x,previous):
		""">> PROCESS_IMAGE: ensures that the image is valid."""
		x = x/255; x -= 0.5
		return x*4*previous.std()+previous.mean()

	def visualize(self,savePath):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		plt.imshow(self.actMax)
		plt.show()
		img = kerasImage.array_to_img(self.actMax,scale=False)
		img.save(savePath,dpi=250)
		vrb.print_msg('========== DONE ==========\n')

	def produce_gif(self,savePath):
		""">> PRODUCE_GIF: produce a gif with all the images recollected"""
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
		""">> PRODUCE_MOSAIC: produce a mosaic with an evolution of the convergency."""
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