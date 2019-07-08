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

class ActMaximization():
	""">> CLASS:ACTMAXIMIZATION: http://arxiv.org/abs/1312.6034."""
	def __init__(self,model,cls):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		self.model = model_remove_softmax(model)
		self.imgInput = self.model.inputs[0]
		self.output = self.model.outputs[0]
		self.numClass = self.output.shape[-1]
		self.size = self.imgInput.shape[1]
		loss = self.output[0,cls]
		grads = K.gradients(loss,self.imgInput)[0]
		self.gradient = K.function([self.imgInput],[loss,grads])
		self.imgData = np.random.normal(0,10,(1,self.size,self.size,3))
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,epochs,learnRate=12000,l2Decay=0.0,
			medFiltSize=5,medFiltEvery=4,earlyStop=0,blurStd=0.12,
			blurEvery=20,momentum=0.9,verbose=True):
		""">> EXECUTE: returns the result of the method."""
		velocity = np.zeros(self.imgInput.shape[1:])
		self.gifImg = []
		self.gifImg.append(self.imgData[0].copy())
		for k in range(epochs):
			lossVal,gradVal = self.gradient([self.imgData+momentum*velocity])
			velocity = momentum*velocity+learnRate*gradVal
			self.imgData += velocity
			vrb.print_msg('Current loss value: '+str(lossVal))
			if earlyStop is not 0 and lossVal >= earlyStop:
				vrb.print_msg('Early Stopping achieved on epoch: '+str(k))
				break
			self.imgData = np.clip(self.imgData,0,255).astype('float32')
			if k != epochs-1:
				if l2Decay > 0:
					self.imgData *= (1-l2Decay)
				if blurStd is not 0 and k % blurEvery == 0:
					self.imgData = gaussian_filter(self.imgData, sigma=[0, blurStd, blurStd, 0])
				if medFiltSize is not 0 and k % medFiltEvery == 0 :
					self.imgData = median_filter(self.imgData, size=(1, medFiltSize, medFiltSize, 1))
			self.gifImg.append(self.imgData[0].copy())
		img = deprocess_image(self.imgData[0])
		self.actMax = img
		vrb.print_msg('========== DONE ==========\n')
		return self.actMax

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
		""">> PRODUCE_GIF: with the images recolected generate a gif."""
		vrb.print_msg('Produce '+self.__class__.__name__+' gif...')
		vrb.print_msg('--------------------------')
		size = int(self.size/2) 
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
		stop = False
		size = int(self.size/2) 
		mosaic = []
		for im in self.gifImg[0::samples]:
			image = deprocess_image(im.copy())
			mosaic.append(np.asarray(pilImage.fromarray(image).resize((size,size),pilImage.ANTIALIAS)))
		n = int(np.round(np.sqrt(len(mosaic))))
		cols = size*n+(n-1)*margin
		rows = size*n+(n-1)*margin
		draw = np.zeros((cols,rows,3),dtype='uint8')
		im = 0
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





