# -- DATA UTILITIES -- #
from __future__ import absolute_import

# -- IMPORTS -- #
from .. import print_msg
from keras.preprocessing import image as kerasImage
from PIL import Image as pilImage
from glob import glob
import keras.backend as K
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# >> NORM_IMG: Normalizes image data.
def norm_img(img, interval=[0,1], dtype='float32'):
	assert len(interval) == 2
	magnitude = interval[1]-interval[0]
	return (magnitude*(img-np.min(img)))/(np.ptp(img).astype(dtype)+a[0])

# >> STD_IMG: Standarize image data.
def std_img(img):
	return (img-np.mean(img))/np.std(img)

# >> IMG_TO_VECTOR: Converts one image into one dimensional vector. 
def img_to_vector(img, linear=True):
	img = np.asarray(img, dtype='float32')
	H,W,Z = img.shape
	if linear:
		R = img[:,:,0].reshape(H*W,1)
		G = img[:,:,1].reshape(H*W,1) 
		B = img[:,:,2].reshape(H*W,1)
		vector = np.concatenate([R,G,B], axis=0)
	else:
		vector = img.reshape(H*W*Z,1)
	assert vector.shape[0]==H*W*Z, print_msg('Vector must be of size (%s,%s)' % (str(H*W*Z),str(1)),show=False,option='error')
	return vector,H,W

# >> VECTOR_TO_IMG: Converts one dimensional vector into an image.
def vector_to_img(array, height, width):
	return array.reshape(height,width,3)

# >> LOAD_IMAGE: Loads an image given the path and sets the size.
def load_image(imgPath, targetSize=(224,224), preprocess=True):
	data = kerasImage.load_img(imgPath, target_size=targetSize)
	if preprocess:
		data = kerasImage.img_to_array(data)
		data = np.expand_dims(data, axis=0)
	return data

# >> GET_IMAGE_LIST: Get a list with all image filename/s of a certain directory
def get_image_list(dirPath,imgFormat,justOne=True):
	cwd = os.getcwd()
	fullPath = cwd+os.sep+dirPath+os.sep+'*.'+imgFormat
	fileNames = glob(fullPath)
	if justOne:
		imgSel = random.randint(0, len(fileNames)-1)
		return fileNames[imgSel]
	else:
		return fileNames

# >> REDUCE_CHANNELS: Converts a three channel image to just one channel.
def reduce_channels(imgData,axis=-1,option='sum'):
	if option == 'sum':
		return imgData.sum(axis=axis)
	elif option == 'mean':
		return imgData.mean(axis=axis)
	else:
		raise NotImplementedError

# >> DEPROCESS_IMAGE: Converts an image into a valid image.
def deprocess_image(img,scale=0.1,dtype='uint8'):
	img -= img.mean(); img /= (img.std()+K.epsilon())
	img *= scale; img += 0.5; img = np.clip(img,0,1)
	img *= 255; img = np.clip(img,0,255).astype(dtype)
	return img

def visualize_heatmap(image,heatmap,modelName,cmap,savePath):
    fig = plt.figure(figsize=(6, 4))
    plt.subplot(121)
    plt.title('Raw Image')
    plt.axis('off')
    plt.imshow(image)
    plt.subplot(122)
    plt.title(modelName)
    plt.axis('off')
    plt.imshow(heatmap,interpolation='bilinear',cmap=cmap)
    fig.savefig(savePath,dpi=250)