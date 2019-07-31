from __future__ import absolute_import

# -- EXTERN IMPORT -- #
import PIL.Image as pilImage
import numpy as np
import keras.backend as K

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image,visualize_heatmap
from ..utils.bases.layers import DConv2D,DInput,DFlatten,DActivation,DPooling,DDense,DBatch
from ..utils.interfaces import Method
from ..utils.exceptions import NotAValidTensorError, OptionNotSupported

class Deconvolution(Method):
	"""CLASS::Deconvolution:
		---
		Description:
		---
		> Reconstructs the activation from a selected layer.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- layerName {string} -- The name of the layer to reconstruct its activation.
		Raises:
		---
		>- NotAValidTensorError {Exception} -- If the layer specified can not be handled by Deconvolution.
		Link:
		---
		>- http://arxiv.org/abs/1311.2901."""
	def __init__(self,model,layerName):
		self.numFilters = model.get_layer(layerName).output_shape[3]
		self.deconvLayers = []
		self.layerName = layerName
		for i in range(len(model.layers)):
			if model.layers[i].__class__.__name__ == 'Conv2D':
				self.deconvLayers.append(DConv2D(model.layers[i]))
				self.deconvLayers.append(DActivation(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'MaxPooling2D':
				self.deconvLayers.append(DPooling(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'Dense':
				self.deconvLayers.append(DDense(model.layers[i]))
				self.deconvLayers.append(DActivation(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'Activation':
				self.deconvLayers.append(DActivation(model.alyers[i]))
			elif model.layers[i].__class__.__name__ == 'Flatten':
				self.deconvLayers.append(DFlatten(model.layers[i]))
			elif model.layers[i].__class__.__name__ == 'InputLayer':
				self.deconvLayers.append(DInput(model.layers[i]))
			else:
				raise NotAValidTensorError('The layer ['+model.layers[i].name+'] can not be handled by "'+
												self.__class__.__name__+'" method.')
			if self.layerName == model.layers[i].name:
				break
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName,featVis,visMode='all'):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path of the image data.
			>- featVis {int} -- feature Map to visualize. 
			>- visMode {string} -- The mode of reconstruction. (default:{'all'}).
			>>- 'all'
			>>- 'max'
			Returns:
			---
			>- {np.array} -- Image representing the reconstruction of the activation.
			Raises:
			---
			>- ValueError {Exception} -- If the output has wrong dimensions.
			>- OptionNotSupported {Exception} -- If 'visMode' option chosed is not supported."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		if not 0 <= featVis <= self.numFilters
		self.rawData, imgInput = load_image(fileName,preprocess=True)
		vrb.print_msg('Forward Pass')
		vrb.print_msg('--------------------------')
		self.deconvLayers[0].up(imgInput)
		for k in range(1, len(self.deconvLayers)):
			self.deconvLayers[k].up(self.deconvLayers[k - 1].upData)
		output = self.deconvLayers[-1].upData
		if output.ndim != 2 or output.ndim != 4:
			raise ValueError('The output dimensions from the layer "'+
								self.deconvLayers[-1].layer.name+'" is not valid.')
		if output.ndim == 2:
			featureMap = output[:,featVis]
		else:
			featureMap = output[:,:,:,featVis]
		if visMode == 'max':
			maxAct = featureMap.max()
			temp = featureMap == maxAct
			featureMap = featureMap * temp
		elif visMode != 'all':
			raise OptionNotSupported('The option "'+visMode+'" is not supported.')
		output = np.zeros_like(output)
		if 2 == output.ndim:
			output[:,featVis] = featureMap
		else:
			output[:,:,:,featVis] = featureMap
		vrb.print_msg('Backward Pass')
		vrb.print_msg('--------------------------')
		self.deconvLayers[-1].down(output)
		for k in range(len(self.deconvLayers)-2,-1,-1):
			self.deconvLayers[k].down(self.deconvLayers[k + 1].downData)
		deconv = self.deconvLayers[0].downData
		self.deconv = deconv.squeeze()
		vrb.print_msg('========== DONE ==========\n')
		return self.deconv

	def visualize(self,savePath):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- The path where the graph will be saved.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize'+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		result = (self.deconv-self.deconv.min())/(self.deconv.max()-self.deconv.min()+K.epsilon())
		uint8Deconv = (result* 255).astype(np.uint8)
		img = pilImage.fromarray(uint8Deconv, 'RGB')
		visualize_heatmap(self.rawData,img,self.__class__.__name__,'viridis',savePath)
		vrb.print_msg('========== DONE ==========\n')
