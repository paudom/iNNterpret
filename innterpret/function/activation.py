from __future__ import absolute_import

# -- EXTERN IMPORT -- #
from keras.models import Model
import keras.backend as K
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import os

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import load_image
from ..utils.tensor import get_conv_layers
from ..utils.interfaces import Method

class ActivationVis(Method):
	"""CLASS::ActivationVis: 
		---
		Description:
		---
		> Method that allows you to get the activations from all layers.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- saveDir {string} -- directory path where the images will be saved.
		Raises:
		---
		>- NotADirectoryError {Exception} -- If the directory introduced does not exists."""
	def __init__(self,model,saveDir):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------\n')
		if not os.path.isdir(saveDir):
			raise NotADirectoryError('['+saveDir+'] is not a directory.')
		self.saveDir = saveDir
		self.layerNames,self.layerOutputs,_ = get_conv_layers(model)
		self.model = Model(inputs=model.input,outputs=self.layerOutputs)
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,filePath,cols=32,getAll=True):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- filePath {string} -- The path of an image file.
			>- cols {int} -- The number of feature maps in a line. (default:{32}).
			>- getAll {bool} -- Flag to get the activations of all images in derectory or not. (default:{True}).
			Returns:
			---
			>- A graph with all the feature maps.
			Raises:
			---
			>- ValueError {Exception} -- If the layer and feature maps selected are invalid."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		imgData = load_image(filePath)
		outputs = self.model.predict(imgData)
		if getAll:
			for n,act in enumerate(outputs):
				numFilters = act.shape[-1]
				rows = numFilters // cols
				fig = plt.figure(figsize=(cols,rows))
				for k in range(0,cols*rows):
					image = act[0,:,:,k]
					image = (image-np.min(image))/(np.max(image)-np.min(image)+K.epsilon())
					fig.add_subplot(rows,cols,k+1)
					plt.imshow(image,cmap='gray')
					plt.xticks([]); plt.yticks([])
				fileName = self.saveDir+os.sep+self.layerNames[n]+'.png'
				fig.savefig(fileName,dpi=250)
		else:
			layer = int(input(vrb.set_msg('Select a layer, from (0-%s): ' % str(len(self.layerNames)))))
			if not 0 <= layer <= len(self.layerNames):
				raise ValueError('The layer introduced is not valid. It has to be between [0,'+
									self.layerNames+'].')
			featureMap = int(input(vrb.set_msg('Select desired feature map, from (0-%s): ' \
				 % str(self.layerOutputs[layer].shape[3]-1))))
			if not 0 <= featureMap <= self.layerOutputs[layer].shape[3]-1:
				raise ValueError('The feature map introduced is not valid. It has to be between [0,'+
									self.layerOutputs[layer].shape[3]-1+'].')
			fig = plt.figure(figsize=(6,4))
			plt.imshow(outputs[layer][0,:,:,featureMap],cmap='gray')
			plt.xticks([]); plt.yticks([])
			fileName = self.saveDir+os.sep+self.layerNames[layer]+'_'+str(featureMap)+'.png'
			fig.savefig(fileName,dpi=250)
		vrb.print_msg('========== DONE ==========\n')
