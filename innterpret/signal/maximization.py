from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
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

# -- ACTIVATION MAXIMIZATION METHOD -- #
class ActMaximization():
	def __init__(self,model,cls):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------\n')
		self.model = model_remove_softmax(model)
		self.imgInput = self.model.inputs[0]
		self.output = self.model.outputs[0]
		self.numClass = self.output.shape[-1]
		self.size = self.imgInput.shape[1]
		loss = self.output[0,selClass]
		grads = K.gradients(loss,self.imgInput)[0]
		self.gradient = K.function([self.imgInput],[loss,grads])
		self.imgData = np.random.normal(0,10,(1,self.size,self.size,3))

	# >> EXECUTE: returns the result of the GradCAM method
	def execute(self,epochs,learnRate=12000,l2Decay=0.0,
			medFiltSize=5,medFiltEvery=4,earlyStop=0,blurStd=0.12,
			blurEvery=20,momentum=0.9,verbose=True):
		velocity = np.zeros(self.imgInput.shape[1:])
		self.gifImg = []
		self.gifImg.append(self.imgData[0].copy())
		for k in range(epochs):
        	lossVal,gradVal = self.gradient([self.imgData+momentum*velocity])
        	velocity = momentum*velocity+learnRate*gradVal
        	self.imgData += velocity
    		if verbose:
        		print_msg('Current loss value: '+str(lossVal),option='verbose')
    		if earlyStop is not 0 and lossVal >= earlyStop:
        		if verbose:
        			print_msg('Early Stopping achieved on epoch: '+str(k),option='verbose')
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
		print_msg('========== DONE ==========\n')
		return self.actMax

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		plt.imshow(self.actMax)
		plt.show()
		img = kerasImage.array_to_img(self.actMax,scale=False)
		img.save(savePath,dpi=250)
		print_msg('========== DONE ==========\n')

	# >> PRODUCE_GIF: produce a gif with all the images recollected
	def produce_gif(self,savePath):
		print_msg('Produce '+self.__class__.__name__+' gif...')
		print_msg('--------------------------')
		size = int(self.targetSize/2) 
		with imageio.get_writer(savePath, mode='I') as writer:
    		for im in self.gifImg:
        		image = deprocess_image(im.copy())
        		image = np.asarray(pilImage.fromarray(image).resize((size,size),pilImage.ANTIALIAS))
        		writer.append_data(image)
        print_msg('========== DONE ==========\n')
	
	# >> PRODUCE_MOSAIC: produce a mosaic with an evolution of the convergency.
    def produce_mosaic(self,samples,savePath):
    	print_msg('Produce '+self.__class__.__name__+' mosaic...')
		print_msg('--------------------------')
    	margin = 5
		stop = False
		size = int(self.targetSize/2) 
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
		print_msg('========== DONE ==========\n')

# -- FEATURE VISUALIZATION METHOD -- #
class FeatMaximization():
	def __init__(self,model,layerName,filt,targetSize):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------\n')
		self.factor = 1.2; self.upSteps = 9
		self.model = model
		self.imgInput = self.model.inputs[0]
		self.layerName = layerName
		assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		self.layer = self.model.get_layer(self.layerName)
		numFilters = self.layer.shape[-1]
		assert 0 <= filt <= numFilters-1
		loss = K.mean(layer[:,:,:,filt])
		grads = K.gradients(loss,self.imgInput)[0]
		grads /= K.sqrt(K.mean(K.square(grads)))+K.epsilon()
		self.gradient = K.function([imgInput],[loss,grads])
		self.targetSize = targetSize
		self.size = int(self.targetSize/(self.factor**self.upSteps))
		self.imgData = np.random.normal(0,10,(1,size,size,3))
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of the GradCAM method
	def execute(self,epochs,verbose=True):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		self.gifImg = []
		self.gifImg.append(self.imgData[0].copy())
		for up in reversed(range(self.upSteps)):
    		for k in range(epochs):
        		lossVal,gradVal = self.gradient([self.imgData])
        		if lossVal <= K.epsilon():
        			if verbose:
            		print_msg('Gradient got stuck',option='verbose')
            	break
        	step = 1/(gradVal.std()+K.epsilon())
        	self.imgData += gradVal*step
        	gifImg.append(self.imgData[0].copy())
        	if verbose:
        		print_msg('Current loss value: '+str(lossVal),option='verbose')
    		size = int(self.targetSize/(self.factor**up))
    		img = deprocess_image(self.imgData[0],scale=0.25)
    		img = np.asarray(pilImage.fromarray(img).resize((size,size),pilImage.BILINEAR),dtype='float32')
    		self.imgData = [self.process_image(img,self.imgData[0])]
		img = deprocess_image(imgInputData[0])
		self.actMax = img
		print_msg('========== DONE ==========\n')
		return self.actMax

	# >> PROCESS_IMAGE: ensures that the image is valid
	def process_image(x,previous):
    	x = x/255; x -= 0.5
    	return x*4*previous.std()+previous.mean()

   	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		plt.imshow(self.actMax)
		plt.show()
		img = kerasImage.array_to_img(self.actMax,scale=False)
		img.save(savePath,dpi=250)
		print_msg('========== DONE ==========\n')

	# >> PRODUCE_GIF: produce a gif with all the images recollected
	def produce_gif(self,savePath):
		print_msg('Produce '+self.__class__.__name__+' gif...')
		print_msg('--------------------------')
		size = int(self.targetSize/2) 
		with imageio.get_writer(savePath, mode='I') as writer:
    		for im in self.gifImg:
        		image = deprocess_image(im.copy())
        		image = np.asarray(pilImage.fromarray(image).resize((size,size),pilImage.ANTIALIAS))
        		writer.append_data(image)
        print_msg('========== DONE ==========\n')
	
	# >> PRODUCE_MOSAIC: produce a mosaic with an evolution of the convergency.
    def produce_mosaic(self,samples,savePath):
    	print_msg('Produce '+self.__class__.__name__+' mosaic...')
		print_msg('--------------------------')
    	margin = 5
		stop = False
		size = int(self.targetSize/2) 
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
		print_msg('========== DONE ==========\n')





