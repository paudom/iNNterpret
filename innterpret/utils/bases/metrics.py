from __future__ import absolute_import

# -- IMPORT -- #
import math
import numpy as np

# -- DISTANCE METRICS -- #
class Metrics():
	""">> CLASS:METRICS: Base that allows to compute different distance operations"""
	def euclidean_distance(self,x,y):
		""">> EUCLIDEAN_DISTANCE: returns the euclidean distance between two arrays."""
		return np.sqrt(np.sum(np.power(np.subtract(x,y),2),axis=-1))

	def manhattan_distance(self,x,y):
		""">> MANHATTAN_DISTANCE: return manhattan distance between two arrays."""
		return np.sum(np.abs(np.subtract(x,y)),axis=-1)
	
	def minkowski_distance(self,x,y,pVal):
		""">> MINKOWSKI_DISTANCE: return minkowski distance between two arrays."""
		return self.nth_root(np.sum(np.power(np.abs(np.subtract(x,y)),pVal),axis=-1),pVal)

	def nth_root(self,value,nRoot):
		""">> NTH_ROOT: return the nth_root of a array."""
		return np.round(value**(1/float(nRoot)))
	
	def cosine_distance(self,x,y):
		""">> COSINE_DISTANCE: return cosine distance between two arrays."""
		y = y.reshape(x.shape[1],x.shape[0])
		return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

	def jaccard_distance(self,x,y):
		""">> JACCARD_DISTANCE: return jaccard distance between two arrays."""
		x = np.asarray(x, np.bool) 
		y = np.asarray(y, np.bool) 
		return np.double(np.bitwise_and(x, y).sum())/np.double(np.bitwise_or(x, y).sum())
