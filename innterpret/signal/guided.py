from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image, deprocess_image, visualize_heatmap
from ..utils.tensor import load_vgg16, load_vgg19, load_model
from ..utils.interfaces import Method
from ..utils.exceptions import NotAConvLayerError
from tensorflow.python.framework import ops
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import keras.backend as K

class GuidedBackProp(Method):
	"""CLASS::GuidedBackProp:
		---
		Description:
		---
		> Eliminate negative influences when computing back propagation.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- Selected layer to visualize.
		>- option {string} -- Model to load. (default:{'vgg16'})
		>>- 'vgg16'
		>>- 'vgg19'
		>>- 'other'.
		>- h5file {string} -- Path to the h5file, specify if option is 'other'.(default:{None}).
		Link:
		---
		>- http://arxiv.org/abs/1412.6806."""
	def __init__(self,model,layerName,option='vgg16',h5file=None):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		if model.get_layer(layerName).__class__.__name__ != 'Conv2D':
			raise NotAConvLayerError('The layer "'+layerName+'" is not a convolution layer.')
		if 'GuidedBackProp' not in ops._gradient_registry._registry:
			@ops.RegisterGradient("GuidedBackProp")
			def _GuidedBackProp(op, grad):
				dtype = op.inputs[0].dtype
				return grad * tf.cast(grad > 0., dtype) * tf.cast(op.inputs[0] > 0., dtype)
		g = tf.get_default_graph()
		with g.gradient_override_map({'Relu':'GuidedBackProp'}):
			layerList = [layer for layer in model.layers[1:] if hasattr(layer, 'activation')]
			for layer in layerList:
				if layer.activation.__name__ == 'relu':
					layer.activation = tf.nn.relu
			if option == 'vgg19':
				guidedModel = load_vgg19()
			elif option == 'vgg16':
				guidedModel = load_vgg16()
			else:
				guidedModel = load_model(h5file)
			modelInput = guidedModel.input
			layerOutput = guidedModel.get_layer(layerName).output
			argument = K.gradients(layerOutput,modelInput)[0]  
			self.gradient = K.function([modelInput],[argument])
			self.model = guidedModel
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- file path to the image data.
			Returns:
			---
			>- {np.array} -- The saliency image."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData, imgInput = load_image(fileName,preprocess=True)
		self.saliency = self.gradient([imgInput])
		self.saliency = self.saliency[0][0,:,:,:]
		vrb.print_msg('========== DONE ==========\n')
		return self.saliency

	def visualize(self,savePath):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- Path where the graph will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize'+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = deprocess_image(self.saliency.copy())
		visualize_heatmap(self.rawData,result,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')