from __future__ import absolute_import

# -- IMPORT -- #
from .. import print_msg
from ..utils.data import get_image_list, load_image
from ..utils.bases import Metrics
import numpy as np
import os

# -- DISTANCE ROBUSTNESS METHOD -- #
class DistRobust():
	def __init__(self,model,classOneDir,classTwoDir):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------')
		self.model = model
		self.metric = Metrics()
		imgFormat = input(print_msg('Select the extension of the images: ',show=False,option='input'))
		if not os.path.isdir(classOneDir):
			assert False, print_msg(classOneDir+' is not a directory.',show=False,option='error')
		if not os.path.isdir(classTwoDir):
			assert False, print_msg(classTwoDir+' is not a directory.',show=False,option='error')
		self.classOne = get_image_list(classOneDir,imgFormat,justOne=False)
		self.classTwo = get_image_list(classTwoDir,imgFormat,justOne=False)
		if self.classOne and self.classTwo:
			print_msg('========== DONE ==========\n')
		else:
			assert False, print_msg('This directory does not contain files with the ['+imgFormat+'] extension.',show=False,option='error')

	# >> EXECUTE: returns the result of the LRP method
	def execute(self,metric):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		oneVect = []; twoVect = []
		for k in range(len(self.classOne)):
			img = load_image(self.classOne[k])
			oneVect.append(self.model.predict(img))
		for k in range(len(self.classTwo)):
			img = load_image(self.classTwo[k])
			twoVect.append(self.model.predict(img))
		meanOne = np.mean(np.array(oneVect),axis=0)
		meanTwo = np.mean(np.array(twoVect),axis=0)
		if metric == 'euclidean':
			result = self.metric.euclidean_distance(meanOne,meanTwo)
		elif metric == 'manhattan':
			result = self.metric.manhattan_distance(meanOne,meanTwo)
		elif metric == 'minkowski':
			result = self.metric.minkowski_distance(meanOne,meanTwo)
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
		print_msg('========== DONE ==========\n')
		return result

# -- SIMILAR DISTANCE EXAMPLES METHOD -- #
class SDEModel():
	def __init__(self,model,directory):
		print_msg(self.__class__.__name__+' Initializing')
		print_msg('--------------------------')
		self.model = model
		self.data = {}
		self.metric = Metrics()
		imgFormat = input(print_msg('Select the extension of the images: ',show=False,option='input'))
		fileNames = get_image_list(directory, imgFormat)
		if fileNames:
			for k in range(len(fileNames)):
				imgArray = load_image(fileNames[k])
				self.data[fileNames[k]] = model.predict(imgArray)
			print_msg('========== DONE ==========\n')
		else:
			assert False, print_msg('This directory does not contain files with the ['+imgFormat+'] extension.',show=False,option='error')

	# >> EXECUTE: returns the result of the LRP method
	def execute(self,fileName,metric,numK):
		print_msg(self.__class__.__name__+' Analyzing')
		print_msg('--------------------------')
		distVector = []
		if metric == 'minkowski':
			pVal = int(input(print_msg('Select the P value for the Minkowski distance: ',show=False,option='input')))
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
				distVector.append(self.metric.minkowski_distance(pred,allOutputs[k],pVal))
			elif metric == 'cosine':
				distVector.append(self.metric.cosine_distance(pred,allOutputs[k]))
			elif metric == 'jaccard':
				distVector.append(self.metric.jaccard_distance(pred,allOutputs[k]))
			else:
				raise NotImplementedError
		indices = np.argsort(np.array(distVector))[:numk]
		candidates = []
		for k in range(len(indices)):
			candidates.append(allFiles[indices[k]])
		print_msg('========== DONE ==========\n')
		return candidates




