from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image
from ..utils.tensor import decode_predictions
from ..utils.interfaces import Method
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import PIL
import numpy as np
import math

class OcclusionMap(Method):
	"""CLASS::OcclusionMap: 
		---
		Description:
		---
		> Method that occludes part of the image and tracks the precision obtained each time.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze."""
	def __init__(self,model):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.inputSize = model.inputs[0].get_shape()[1]
		self.numClasses = model.outputs[0].get_shape()[1]-1
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName,winSize=15,winStride=3):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- path of the image data.
			>- winSize {int} -- The size of the mask. (default:{15}).
			>- winStride {int} -- The stride taken each iteration. (default:{3}).
			Returns:
			>- {np.array} -- A heat map with the important areas of the image."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		#assert 1 <= winSize <= int(self.inputSize/4)
		#assert 1 <= winStride <= 10
		self.rawData = load_image(fileName,preprocess=False)
		self.imgInput = load_image(fileName)
		vrb.print_msg('Preparing Mask')
		vrb.print_msg('--------------------------')
		H,W,_ = self.imgInput[0].shape
		outH = int(math.ceil((H-winSize)/(winStride+1)))
		outW = int(math.ceil((W-winSize)/(winStride+1)))
		heatMap = np.zeros((outH,outW))
		fullPred = self.model.predict(self.imgInput)
		predictions = decode_predictions(fullPred)
		vrb.print_msg('Top 5 predicted classes: '+str(predictions))
		self.maxClass = int(input(vrb.set_msg('Select the class to explain (0-'+str(self.numClasses)+'): ')))
		vrb.print_msg('Prediction of the Selected Class: '+str(fullPred[0][self.maxClass]))
		vrb.print_msg('Number of iterations: '+str(outH*outW))
		vrb.print_msg('Dimensions of Resulting HeatMap: '+str(outH)+'x'+str(outW))
		k = 1
		vrb.print_msg('------------------')
		for row in range(outH):
			for col in range(outW):
				startH = row*winStride; endH = min(H,startH+winSize)
				startW = col*winStride; endW = min(W,startW+winSize)
				imgData = self.imgInput.copy()
				imgData[:,startH:endH,startW:endW,:] = 0
				prediction = self.model.predict(imgData)
				heatMap[row,col] = prediction[0][self.maxClass]
				if k % 50 == 0:
					vrb.print_msg('Result: '+str(self.maxClass)+' ['+str(prediction[0][self.maxClass])+']')
					vrb.print_msg(str(k)+' out of '+str(outH*outW))
					vrb.print_msg('------------------')
				k += 1
		self.invHeatMap = 1-heatMap
		self.outH = outH; self.outW = outW
		vrb.print_msg('========== DONE ==========\n')
		return self.invHeatMap

	def visualize(self,savePath,cmap='jet'):
		"""METHOD::VISUALIZE
			---
			Arguments:
			---
			>- savePath {string} -- The path where the graph will be saved.
			>- cmap {string} -- the color map used to graph the resulting heat map. (default:{'jet'}).
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		imgRes = self.rawData.resize((self.outW, self.outH), PIL.Image.BILINEAR)
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
		vrb.print_msg('========== DONE ==========\n')

