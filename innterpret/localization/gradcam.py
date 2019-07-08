from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.tensor import decode_predictions
from tensorflow.python.framework import ops
from PIL import Image as pilImage
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

class GradCAM():
	""">> CLASS:GRADCAM: http://arxiv.org/abs/1610.02391."""
	def __init__(self,model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		self.model = model
		self.layerName = layerName
		#assert self.model.get_layer(self.layerName).__class__.__name__ == 'Conv2D'
		self.numClasses = model.outputs[0].get_shape()[1]-1
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName,topCls=5,negGrad=False):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgInput = load_image(fileName)
		decoded = decode_predictions(self.model.predict(imgInput),top=topCls)
		vrb.print_msg('Predicted classes: '+str(decoded))
		clSel = int(input(vrb.set_msg('Select the class to explain (0-'+str(self.numClasses)+'): ')))
		clScore = self.model.output[0, clSel]
		convOutput = self.model.get_layer(self.layerName).output
		grads = K.gradients(clScore, convOutput)[0]
		if negGrad:
			vrb.print_msg('Negative Explanation')
			grads = -grads
		vrb.print_msg('Computing HeatMap')
		vrb.print_msg('--------------------------')
		self.gradient = K.function([self.model.input],[convOutput, grads])
		output, gradsVal = self.gradient([imgInput])
		output, gradsVal = output[0, :], gradsVal[0, :, :, :]
		weights = np.mean(gradsVal,axis=(0,1))
		cam = np.dot(output,weights)
		cam = np.asarray(pilImage.fromarray(cam).resize((224,224),pilImage.BILINEAR),dtype='float32')
		cam = np.maximum(cam, K.epsilon())
		self.cam = cam/cam.max()
		self.selectedClass = clSel
		vrb.print_msg('========== DONE ==========\n')
		return self.cam

	def visualize(self,savePath,cmap='jet'):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		fig = plt.figure(figsize=(6, 4))
		plt.subplot(121)
		plt.title('Raw Image')
		plt.axis('off')
		plt.imshow(self.rawData)
		plt.subplot(122)
		plt.title('Oclussion HeatMap for '+str(self.selectedClass))
		plt.axis('off')
		plt.imshow(self.rawData)
		plt.imshow(self.cam,cmap=cmap,interpolation='bilinear',alpha=0.5)
		fig.savefig(savePath,dpi=250)
		vrb.print_msg('========== DONE ==========\n')