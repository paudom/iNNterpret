from __future__ import absolute_import

# -- EXTERN IMPORT -- #
import numpy as np
import os

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.data import get_image_list, load_image
from ..utils.bases.metrics import Metrics, availableMetrics
from ..utils.interfaces import Method
from ..utils.exceptions import EmptyDirectoryError, OptionNotSupported

class SDEModel(Method):
	"""CLASS::SDEModel:
		--- 
		Description:
		---
		> Method that presents similar training examples.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- directory {string} -- Path to the images files to code them.
		Raises:
		---
		>- NotADirectoryError {Exception} -- If the directory introduced is not a directory.
		>- EmptyDirectory {Exception} -- If the directory does not contain files with the specified image format."""
	def __init__(self,model,directory):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.data = {}
		self.metric = Metrics()
		if not os.path.isdir(directory):
			raise NotADirectoryError('The directory "'+directory+'" is not a directory.')
		imgFormat = input(vrb.set_msg('Select the extension of the images: '))
		self.fileNames = get_image_list(directory, imgFormat)
		if self.fileNames:
			for img in fileNames:
				imgArray = load_image(img)
				self.data[img] = model.predict(imgArray)
			vrb.print_msg('========== DONE ==========\n')
		else:
			raise EmptyDirectoryError('The directory "'+
				directory+'" does not contain files with the ['+imgFormat+'] extension.')
			

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
			Raises:
			---
			>- OptionNotSupported {Exception} -- If the metric option is not supported.
			Returns:
			---
			>- {list[string]} -- With all the paths to the files which are closer."""
		if metric not in availableMetrics:
			raise OptionNotSupported('The option "'+metric+'" is not supported.')
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		if metric == 'minkowski':
			pVal = int(input(vrb.set_msg('Select the P value for the Minkowski distance: ')))
		distVector = {}
		imgInput = load_image(fileName)
		pred = self.model.predict(imgInput)
		for imgPath,imgFile in self.data.items():
			if metric == 'euclidean':
				distVector[imgPath] = self.metric.euclidean_distance(pred,imgFile)
			elif metric == 'manhattan':
				distVector[imgPath] = self.metric.manhattan_distance(pred,imgFile)
			elif metric == 'minkowski':
				distVector[imgPath] = self.metric.minkowski_distance(pred,imgFile,pVal)
			elif metric == 'cosine':
				distVector[imgPath] = self.metric.cosine_distance(pred,imgFile)
			elif metric == 'jaccard':
				distVector[imgPath] = self.metric.jaccard_distance(pred,imgFile)
		candidates = []
		for k in range(numK):
			candidates.append(min(distVector,key=distVector.get))
		vrb.print_msg('========== DONE ==========\n')
		return candidates

	def __repr__(self):
		return super().__repr__()+'Similar Distance Examples>'