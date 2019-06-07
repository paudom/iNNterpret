from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.data import load_image
from ..utils.tensor import decode_predictions
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import PIL
import numpy as np
import math

# -- OCCLUSION MAP METHOD -- #
class OcclusionMap():
	def __init__(self,model):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------')
		self.model = model
		self.inputSize = model.inputs[0].get_shape()[1]
		self.numClasses = model.outputs[0].get_shape()[1]-1
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of the Occlusion Map method
	def execute(self,fileName,occSize=15,occStride=3,verbose=True):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		assert 1 <= occSize <= int(self.inputSize/4)
		assert 1 <= occStride <= 10
		self.rawData = load_image(fileName,preprocess=False)
		self.imgInput = load_image(fileName)
		print_msg('Preparing Mask')
		print_msg('--------------------------')
		H,W,_ = imgInput[0].shape
		outH = int(math.ceil((H-occSize)/(occStride+1)))
		outW = int(math.ceil((W-occSize)/(occStride+1)))
		heatMap = np.zeros((outH,outW))
		fullPred = self.model.predict(imgInput)
		predictions = decode_predictions(fullPred)
		print_msg('Top 5 predicted classes: '+str(predictions),option='input')
		self.maxClass = int(input(print_msg('Select the class to explain (0-'+str(self.numClasses)+'): ',
			show=False,option='input')))
		if verbose:
			print_msg('Prediction of the Selected Class: '+str(fullPred[0][self.maxClass]),option='verbose')
			print_msg('Number of iterations: '+str(outH*outW),option='verbose')
			print_msg('Dimensions of Resulting HeatMap: '+str(outH)+'x'+str(outW),option='verbose')
			k = 1
			print_msg('------------------',option='verbose')
		for row in range(outH):
			for col in range(outW):
				startH = row*occStride; endH = min(H,startH+occSize)
				startW = col*occStride; endW = min(W,startW+occSize)
				imgData = imgInput.copy()
				imgData[:,startH:endH,startW:endW,:] = 0
				prediction = self.model.predict(imgData)
				heatMap[row,col] = prediction[0][self.maxClass]
				if k % 50 == 0 and verbose:
					print_msg('Result: '+str(self.maxClass)+' ['+str(prediction[0][self.maxClass])+']',option='verbose')
					print_msg(str(k)+' out of '+str(outH*outW),option='verbose')
					print_msg('------------------',option='verbose')
				k += 1
		self.invHeatMap = 1-heatMap
		print_msg('========== DONE ==========\n')
		return self.invHeatMap

	# >> VISUALIZE: returns a graph with the results.
	def visualize(self,savePath,cmap='jet'):
		print_msg('Visualize '+self.__class__.__name__+' Result...')
		print_msg('--------------------------')
		imgRes = self.rawData.resize((outW, outH), PIL.Image.BILINEAR)
		fig = plt.figure(figsize=(6, 4))
		plt.subplot(121)
		plt.title('Raw Image')
		plt.axis('off')
		plt.imshow(self.rawData)
		plt.subplot(122)
		plt.title(self.__class__.__name__+' for '+str(self.maxClass))
		plt.axis('off')
		plt.imshow(imgRes)
		plt.imshow(self.invHeatMap,cmap=cmap,interpolation='bilinear',alpha=0.5)
		fig.savefig(savePath,dpi=250)
		print_msg('========== DONE ==========\n')

