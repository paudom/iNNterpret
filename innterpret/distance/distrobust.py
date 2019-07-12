from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.bases.metrics import Metrics
from ..utils.data import get_image_list, load_image
from ..utils.interfaces import Method
import numpy as np
import os

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
		>- classTwoDir {string} -- Path containing the images of another class. """
	def __init__(self,model,classOneDir,classTwoDir):
		vrb.print_msg(self.__class__.__name__+' Initializing')
		vrb.print_msg('--------------------------')
		self.model = model
		self.metric = Metrics()
		imgFormat = input(vrb.set_msg('Select the extension of the images: '))
		#if not os.path.isdir(classOneDir):
			#assert False, print_msg(classOneDir+' is not a directory.',show=False,option='error')
		#if not os.path.isdir(classTwoDir):
			#assert False, print_msg(classTwoDir+' is not a directory.',show=False,option='error')
		self.classOne = get_image_list(classOneDir,imgFormat,justOne=False)
		self.classTwo = get_image_list(classTwoDir,imgFormat,justOne=False)
		if self.classOne and self.classTwo:
			vrb.print_msg('========== DONE ==========\n')
		#else:
			#assert False, print_msg('This directory does not contain files with the ['+imgFormat+'] extension.',show=False,option='error')

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
			Returns:
			---
			>- {float} -- The average distance between the two classes."""
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
		else:
			raise NotImplementedError
		self.vectOne = oneVect
		self.vectTwo = twoVect
		self.meanOne = meanOne
		self.meanTwo = meanTwo
		vrb.print_msg('========== DONE ==========\n')
		return result




