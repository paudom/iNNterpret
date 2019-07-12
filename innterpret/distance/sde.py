from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import get_image_list, load_image
from ..utils.bases.metrics import Metrics
from ..utils.interfaces import Method
import numpy as np
import os

class SDEModel(Method):
	"""CLASS::SDEModel:
		--- 
		Description:
		---
		> Method that presents similar training examples.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- directory {string} -- Path to the images files to code them."""
	def __init__(self,model,directory):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.data = {}
		self.metric = Metrics()
		imgFormat = input(vrb.set_msg('Select the extension of the images: '))
		fileNames = get_image_list(directory, imgFormat)
		if fileNames:
			for k in range(len(fileNames)):
				imgArray = load_image(fileNames[k])
				self.data[fileNames[k]] = model.predict(imgArray)
			vrb.print_msg('========== DONE ==========\n')
		#else:
			#assert False, print_msg('This directory does not contain files with the ['+imgFormat+'] extension.',show=False,option='error')

	def interpret(self,fileName,metric,numK):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path to the image file.
			>- metric {string} -- the distance metric used.
			>>- 'euclidean'
			>>- 'manhattan'
			>>- 'cosine'
			>>- 'minkwoski'
			>>- 'jaccard'
			>- numK {int} -- Number of images to present.
			Returns:
			---
			>- {list[string]} -- With all the paths to the files which are closer."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		distVector = []
		imgInput = load_image(fileName)
		pred = self.model.predict(imgInput)
		allOutputs = list(self.data.values())
		allFiles = self.data.keys()
		for k in range(len(allOutputs)):
			if metric == 'euclidean':
				distVector.append(self.metric.euclidean_distance(pred,allOutputs[k]))
			elif metric == 'manhattan':
				distVector.append(self.metric.manhattan_distance(pred,allOutputs[k]))
			elif metric == 'minkowski':
				pVal = int(input(vrb.set_msg('Select the P value for the Minkowski distance: ')))
				distVector.append(self.metric.minkowski_distance(pred,allOutputs[k],pVal))
			elif metric == 'cosine':
				distVector.append(self.metric.cosine_distance(pred,allOutputs[k]))
			elif metric == 'jaccard':
				distVector.append(self.metric.jaccard_distance(pred,allOutputs[k]))
			else:
				raise NotImplementedError
		indices = np.argsort(np.array(distVector))[:numK]
		candidates = []
		for k in range(len(indices)):
			candidates.append(allFiles[indices[k]])
		vrb.print_msg('========== DONE ==========\n')
		return candidates