from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.data import load_image
from keras.models import Model
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import os

# -- ACTIVATION VISUALIZATION METHOD -- #
class ActivationVis():
	def __init__(self,model,saveDir):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------\n')
		if not os.path.isdir(saveDir):
			assert False, print_msg(saveDir+'is not a directory.',show=False,option='error')
		self.saveDir = saveDir
		layerOutputs = []; layerNames = []
		for layer in model.layers:
			if layer.__class__.__name__ == 'Conv2D':
				layerNames.append(layer.name)
				layerOutputs.append(layer.output)
		if not layerOutputs:
			assert False, print_msg('The model introduced do not have any Conv2D layer',show=False,option='error')
		self.model = Model(inputs=model.input,outputs=layerOutputs)
		self.layerNames = layerNames
		self.layerOutputs = layerOutputs
		print_msg('========== DONE ==========\n')

	# >> EXECUTE: returns the result of the ActivationVis method
	def execute(self,fileName,cols=32,getAll=True):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		imgData = load_image(fileName)
		outputs = self.model.predict(imgData)
		if getAll:
			for n,act in zip(range(len(outputs)),outputs):
				numFilters = act.shape[-1]; size = act.shape[1]
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
			for k in self.layerNames:
				print_msg(layerNames[k]+str(k),option='verbose')
			layer = int(input(print_msg('Select a layer, from (0-%s): ' % str(len(self.layerNames)),show=False,option='input')))
			fmap = int(input(print_msg('Select desired feature map, from (0-%s): ' % str(self.layerOutputs[layer].shape[3]-1),show=False,option='input')))
			fig = plt.figure(figsize=(6,4))
			plt.imshow(outputs[layer][0,:,:,fmap],cmap='gray')
			plt.xticks([]); plt.yticks([])
			fileName = self.saveDir+os.sep+self.layerNames[layer]+'_'+str(fmap)+'.png'
			fig.savefig(fileName,dpi=250)
		print_msg('========== DONE ==========\n')
