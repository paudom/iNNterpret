from __future__ import absolute_import

# -- IMPORT -- #
import math
import numpy as np

# -- AVAILABLE METRICS  -- #
availableMetrics = ['euclidean','manhattan','minkowski','cosine','jaccard']

# -- DISTANCE METRICS -- #
class Metrics():
	"""CLASS::Metrics:
		---
		Description:
		---
		>Base that allows to compute different distance operations."""
	def euclidean_distance(self,x,y):
		"""METHOD::EUCLIDEAN_DISTANCE:
			---
			Arguments:
			---
			>- x {np.array} -- array one.
			>- y {np.array} -- array two.
			Returns:
			---
			>- {np.array} -- The euclidean distance between the two arrays."""
		return np.sqrt(np.sum(np.power(np.subtract(x,y),2),axis=-1))

	def manhattan_distance(self,x,y):
		"""METHOD::MANHATTAN_DISTANCE:
			---
			Arguments:
			---
			>- x {np.array} -- array one.
			>- y {np.array} -- array two.
			Returns:
			---
			>- {np.array} -- The manhattan distance between the two arrays."""
		return np.sum(np.abs(np.subtract(x,y)),axis=-1)
	
	def minkowski_distance(self,x,y,pVal):
		"""METHOD::MINKOWSKI_DISTANCE:
			---
			Arguments:
			---
			>- x {np.array} -- array one.
			>- y {np.array} -- array two.
			>- pVal {int} -- p value for minkowski distance.
			Returns:
			---
			>- {np.array} -- The minkowski distance between the two arrays."""
		return self.__nth_root(np.sum(np.power(np.abs(np.subtract(x,y)),pVal),axis=-1),pVal)
	
	def cosine_distance(self,x,y):
		"""METHOD::COSINE_DISTANCE:
			---
			Arguments:
			---
			>- x {np.array} -- array one.
			>- y {np.array} -- array two.
			Returns:
			---
			>- {np.array} -- The cosine distance between two arrays."""
		y = y.reshape(x.shape[1],x.shape[0])
		return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

	def jaccard_distance(self,x,y):
		"""METHOD::JACCARD_DISTANCE:
			---
			Arguments:
			---
			>- x {np.array} -- array one.
			>- y {np.array} -- array two.
			Returns:
			---
			>- {np.array} -- The jaccard distance between two arrays."""
		x = np.asarray(x, np.bool) 
		y = np.asarray(y, np.bool) 
		return np.double(np.bitwise_and(x, y).sum())/np.double(np.bitwise_or(x, y).sum())
	
	def __nth_root(self,value,nRoot):
		"""METHOD::__NTH_ROOT:
			---
			Arguments:
			---
			>- value {np.array} -- array containing the values to root.
			>- nRoot {int} -- root number.
			Returns:
			---
			>- {np.array} -- The nth_root of a array"""
		return np.round(value**(1/float(nRoot)))

	def __repr__(self):
		return '<class::Metrics -- Distance Computation>'
