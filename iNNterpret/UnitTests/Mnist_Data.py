# -- IMPORTS -- #
from keras.datasets import mnist
from keras.preprocessing import image as kerasImage
from PIL import Image as pilImage
import keras.utils as kerasUtils
import numpy as np
import pickle

# -- DEFINE FUNCTIONS -- #
def split_validation(Xtrain,Ytrain,splitRatio):
	total = Xtrain.shape[0]
	cutTrain = int(round(total*(1-splitRatio), 0))
	trainSet = (Xtrain[0:cutTrain,:,:,:],Ytrain[0:cutTrain,:])
	validSet = (Xtrain[cutTrain:,:,:,:],Ytrain[cutTrain:,:])
	return trainSet,validSet

_,(Xtest,Ytest) = mnist.load_data()
Xtest = np.expand_dims(Xtest,-1)
Xtest = np.asarray([kerasUtils.normalize(
			kerasImage.img_to_array(
			kerasImage.array_to_img(im,scale=False).convert('RGB').resize((32,32))),axis=0) for im in Xtest])
Ytest = kerasUtils.to_categorical(Ytest)
print(Xtest.shape)
print(Ytest.shape)
with open('./Data/Mnist_data.pkl','wb') as file:
	pickle.dump((Xtest,Ytest),file)