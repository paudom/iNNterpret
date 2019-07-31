from __future__ import absolute_import

# -- EXTERN IMPORT -- #
from PIL import Image as pilImage
import numpy as np
import keras.backend as K
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.tensor import decode_predictions
from ..utils.interfaces import Method
from ..utils.exceptions import NotAValidTensorError

class GradCAM(Method):
	"""CLASS::GradCAM:
		---
		Description:
		---
		> Method to visualized where model is centring its attention.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- The selected layer to visualize.
		Raises:
		---
		>- NotAValidTensorError {Exception} -- If the layer specified is not a convolution layer.
		Link:
		---
		>- http://arxiv.org/abs/1610.02391."""
	def __init__(self,model,layerName):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		self.model = model
		if self.model.get_layer(layerName).__class__.__name__ != 'Conv2D':
			raise NotAValidTensorError('The layer "'+layerName+'" is not a convolution layer.')
		self.layerName = layerName
		self.numClasses = model.outputs[0].get_shape()[1]-1
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName,topCls=5,negGrad=False):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path of the image data.
			>- topCls {int} -- The number classes with the highest propability to show. (default:{5}).
			>- negGrad {bool} -- Flag to determine how the gradients are computed. (default:{False}).
			Returns:
			---
			>- {np.array} -- A heat map representing the areas where the model is focussing.
			Raises:
			---
			>- ValueError {Exception} -- If the selected class is not valid."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData,imgInput = load_image(fileName,preprocess=True)
		decoded = decode_predictions(self.model.predict(imgInput),top=topCls)
		vrb.print_msg('Predicted classes: '+str(decoded))
		self.selClass = int(input(vrb.set_msg('Select the class to explain (0-'+
												str(self.numClasses)+'): ')))
		if not 0 <= self.selClass <= self.numClasses:
			raise ValueError('The selected class is not valid. It has to be between [0,'+
								self.numClasses+'].')
		clScore = self.model.output[0, self.selClass]
		convOutput = self.model.get_layer(self.layerName).output
		grads = K.gradients(clScore, convOutput)[0]
		if negGrad:
			vrb.print_msg('Setting Negative Explanation.')
			grads = -grads
		vrb.print_msg('Computing HeatMap')
		vrb.print_msg('--------------------------')
		self.gradient = K.function([self.model.input],[convOutput, grads])
		output, gradsVal = self.gradient([imgInput])
		weights = np.mean(gradsVal[0, :, :, :],axis=(0,1))
		self.cam = np.dot(output[0, :],weights)
		self.cam = np.asarray(pilImage.fromarray(self.cam).resize((224,224),pilImage.BILINEAR),
								dtype='float32')
		self.cam = np.maximum(self.cam, K.epsilon())
		self.cam = self.cam/self.cam.max()
		vrb.print_msg('========== DONE ==========\n')
		return self.cam

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