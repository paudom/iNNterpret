# -- PREPROCESS DATA UTILS -- #
	
# -- IMPORTS -- #
import numpy as np

# -- SPLIT_VALIDATION -- #: Splits the training data into two sets, for validation
def split_validation(Xtrain,Ytrain,splitRatio):
	total = Xtrain.shape[0]
	cutTrain = int(round(total*(1-splitRatio), 0))
	trainSet = (Xtrain[0:cutTrain,:,:,:],Ytrain[0:cutTrain,:])
	validSet = (Xtrain[cutTrain:,:,:,:],Ytrain[cutTrain:,:])
	return trainSet,validSet