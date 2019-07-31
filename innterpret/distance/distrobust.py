from __future__ import absolute_import

# -- EXTERN IMPORT -- #
import numpy as np
import os

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.bases.metrics import Metrics, availableMetrics
from ..utils.data import get_image_list, load_image
from ..utils.interfaces import Method
from ..utils.exceptions import EmptyDirectoryError, OptionNotSupported

class DistRobust(Method):
	"""CLASS::DistRobust:
		---
		Description:
		---
		> Method that computes the distance between two classes.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		>- classOneDir {string} -- Path containing the images of a certain class.
		>- classTwoDir {string} -- Path containing the images of another class.
		Raises:
		---
		>- NotADirectoryError {Exception} -- If the directories are indeed not directories.
		>- EmptyDirectory {Exception} -- If the directory does not contain files with the specified image format."""
	def __init__(self,model,classOneDir,classTwoDir):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.metric = Metrics()
		imgFormat = input(vrb.set_msg('Select the extension of the images: '))
		if not os.path.isdir(classOneDir):
			raise NotADirectoryError('The directory "'+classOneDir+'" is not a directory.')
		if not os.path.isdir(classTwoDir):
			raise NotADirectoryError('The directory "'+classTwoDir+'" is not a directory.')
		self.classOne = get_image_list(classOneDir,imgFormat,justOne=False)
		self.classTwo = get_image_list(classTwoDir,imgFormat,justOne=False)
		if not self.classOne:
			raise EmptyDirectoryError('The directory "'+
				classOneDir+'" does not contain files with the ['+imgFormat+'] extension.')
		if not self.classTwo:
			raise EmptyDirectoryError('The directory "'+
				classTwoDir+'" does not contain files with the ['+imgFormat+'] extension.')
		vrb.print_msg('========== DONE ==========\n')
			
	def interpret(self,metric):
		"""METHOD::INTERPRET:
			---
			Arguments:
			>- metric {string} -- Selects the type of distance you want.
			>>- 'euclidean'
			>>- 'manhattan'
			>>- 'cosine'
			>>- 'minkowski'
			>>- 'jaccard'
			Raises:
			---
			>- OptionNotSupported {Exception} -- If the metric selected is not supported.
			Returns:
			---
			>- {float} -- The average distance between the two classes."""
		if metric not in availableMetrics:
			raise OptionNotSupported('The option "'+metric+'" is not supported.')
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		oneVect = []; twoVect = []
		for image in self.classOne:
			img = load_image(image)
			oneVect.append(self.model.predict(img))
		for image in self.classTwo:
			img = load_image(image)
			twoVect.append(self.model.predict(img))
		meanOne = np.mean(np.array(oneVect),axis=0)
		meanTwo = np.mean(np.array(twoVect),axis=0)
		if metric == 'euclidean':
			result = self.metric.euclidean_distance(meanOne,meanTwo)
		elif metric == 'manhattan':
			result = self.metric.manhattan_distance(meanOne,meanTwo)
		elif metric == 'minkowski':
			pVal = int(input(vrb.set_msg('Select the power for the Minkowski Distance: ')))
			result = self.metric.minkowski_distance(meanOne,meanTwo,pVal)
		elif metric == 'cosine':
			result = self.metric.cosine_distance(meanOne,meanTwo)
		elif metric == 'jaccard':
			result = self.metric.jaccard_distance(meanOne,meanTwo)
		vrb.print_msg('========== DONE ==========\n')
		return result




