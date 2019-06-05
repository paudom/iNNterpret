# -- BASE UTILITIES -- #
from __future__ import absolute_import

# -- IMPORT -- #
from colored import fg,bg,attr
import math
import numpy as np

# -- DISTANCE METRICS -- #
class Metrics():

	# >> EUCLIDEAN_DISTANCE: returns the euclidean distance between two arrays.
	def euclidean_distance(self,x,y):
		return np.sqrt(np.sum(np.pow(np.subtract(x,y),2)),axis=-1)

	# >> MANHATTAN_DISTANCE: return manhattan distance between two arrays.
	def manhattan_distance(self,x,y):
		return np.sum(np.abs(np.subtract(x-y)),axis=-1)

	# >> MINKOWSKI_DISTANCE: return minkowski distance between two arrays.	
	def minkowski_distance(self,x,y,pVal):
		return self.nth_root(np.sum(np.pow(np.abs(np.subtract(x-y)),pVal),axis=-1),pVal)

	# >> NTH_ROOT: return the nth_root of a array.
	def nth_root(self,value,nRoot):
		return np.round(value**(1/float(nRoot)))

	# >> COSINE_DISTANCE: return cosine distance between two arrays.	
	def cosine_distance(self,x,y):
		return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

	# >> JACCARD_DISTANCE: return jaccard distance between two arrays.
	def jaccard_distance(self,x,y):
		x = np.asarray(x, np.bool) 
		y = np.asarray(y, np.bool) 
		return np.double(np.bitwise_and(x, y).sum())/np.double(np.bitwise_or(x, y).sum())
