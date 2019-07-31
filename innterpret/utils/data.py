from __future__ import absolute_import

# -- IMPORTS -- #
from keras.preprocessing import image as kerasImage
from PIL import Image as pilImage
from glob import glob
import keras.backend as K
from utils.exceptions import OptionNotSupported
import matplotlib
matplotlib.use('tkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random
import os

def norm_img(img, interval=[0,1],dtype='float32'):
	"""FUNCTION::NORM_IMG: Normalizes image data.
		---
		Arguments:
		---
		>- img {np.array} -- Image in form of a numpy array.
		>- interval {list} -- List containing the min and max values to normalize. (default:{[0,1]})
		>- dtype {string} -- String indicating the type of the result type. (default:{'float32'})
		Returns:
		---
		>- {np.array} -- The image normalized."""
	#assert len(interval) == 2
	magnitude = interval[1]-interval[0]
	return (magnitude*(img-np.min(img)))/(np.ptp(img).astype(dtype)+K.epsilon())

def std_img(img):
	"""FUNCTION::STD_IMG: Standarize image data.
		---
		Arguments:
		>- img {np.array} -- Image in form of a numpy array.
		Returns:
		---
		>- {np.array} -- The image standarized."""
	return (img-np.mean(img))/(np.std(img)+K.epsilon())
 
def img_to_vector(img, linear=True):
	"""FUNCTION::IMG_TO_VECTOR: Converts one image into one dimensional vector.
		---
		Arguments:
		---
		>- img {np.array} -- Image data.
		>- linear {bool} -- falg to reshape manually (True) or us reshape. (default: {True}).
		Returns:
		---
		>- {np.array} -- A vector of one dimension representing the image."""
	img = np.asarray(img, dtype='float32')
	H,W,Z = img.shape
	if linear:
		R = img[:,:,0].reshape(H*W,1)
		G = img[:,:,1].reshape(H*W,1) 
		B = img[:,:,2].reshape(H*W,1)
		vector = np.concatenate([R,G,B], axis=0)
	else:
		vector = img.reshape(H*W*Z,1)
	#assert vector.shape[0]==H*W*Z, print_msg('Vector must be of size (%s,%s)' % (str(H*W*Z),str(1)),show=False,option='error')
	return vector,H,W

def vector_to_img(array, height, width):
	"""FUNCTION::VECTOR_TO_IMG: Converts one dimensional vector into an image.
		---
		Arguments:
		---
		>- array {np.array} -- one dimensional array.
		>- height {int} -- Indicates the height of the original image.
		>- width {int} -- Indicates the width og the original image.
		Returns:
		---
		>- {np.array} -- A numpy array representing the image reconstructed."""
	return array.reshape(height,width,3)

def load_image(imgPath, targetSize=(224,224), preprocess=False):
	"""FUNCTION::LOAD_IMAGE: Loads an image given the path and sets the size.
		---
		Arguments:
		---
		>- imgPath {string} -- Path of the image.
		>- targetSize {tuple(int,int)} -- Height and Widht of the resulting image. (default:{(224,224)}).
		>- preprocess {bool} -- A flag to get a preprocess or get the unmodified image. (default:{False}).
		Returns:
		---
		>- If 'preprocess' is desactivated then:
		>>- {PIL.image} -- original image.
		>>- {np.array} -- data representing the image.
		>- If not:
		>>- {np.array} -- data representing the image.
		Raises:
		---
		>- FileNotFound {Exception} -- If the filename specified is not found."""
	if not os.path.isfile(imgPath):
		raise FileNotFoundError('The file "'+imgPath+'" is not found.')
	img = kerasImage.load_img(imgPath, target_size=targetSize)
	data = process_image(img)
	if preprocess:
		return img,data
	return data

def process_image(imgData):
	"""FUNCTION::PROCESS_IMAGE: Process an image to prepare it for the model."""
	data = kerasImage.img_to_array(imgData)
	return np.expand_dims(data,axis=0)

def get_image_list(dirPath,imgFormat,justOne=True):
	"""FUNCTION::GET_IMAGE_LIST: Get a list with all image filename/s of a certain directory.
		---
		Arguments:
		---
		>- dirPath {string} -- Path of the directory where the images are.
		>- imgFormat {string} -- Format of the images on that folder.
		>- justOne {bool} -- Extract just one or the entire directory. (default:{True}).
		Returns:
		---
		>- {list[string]} -- List containing all the string paths with the image/s."""
	fullPath = dirPath+os.sep+'*.'+imgFormat
	fileNames = glob(fullPath)
	if justOne:
		imgSel = random.randint(0, len(fileNames)-1)
		return fileNames[imgSel]
	else:
		return fileNames

def reduce_channels(imgData,axis=-1,option='sum'):
	"""FUNCTION::REDUCE_CHANNELS: Converts a three channel image to just one channel.
		---
		Arguments:
		---
		>- imgData {np.array} -- Image data.
		>- axis {int} -- To which axis to apply the reduction. (default:{-1}).
		>- option {string} -- How to reduce the dimension. (default:{'sum'}).
		>>- 'sum'
		>>- 'mean'
		Returns:
		---
		>- {np.array} -- image data with just one channel."""
	if option == 'sum':
		return imgData.sum(axis=axis)
	elif option == 'mean':
		return imgData.mean(axis=axis)
	else:
		raise OptionNotSupported('The option "'+option+'" is not supported.')

def deprocess_image(img,scale=0.1,dtype='uint8'):
	"""FUNCTION::DEPROCESS_IMAGE: Converts an image into a valid image.
		---
		Arguments:
		---
		>- img {np.array} -- Image data.
		>- scale {float} -- Scale at which is multiplied the normalization image. (default:{0.1}).
		>- dtype {string} -- Indicates to which type the data is returned. (default:{'uint8'}).
		Returns:
		---
		>- {np.array} -- Deprocessed image data."""
	img -= img.mean(); img /= (img.std()+K.epsilon())
	img *= scale; img += 0.5; img = np.clip(img,0,1)
	img *= 255; img = np.clip(img,0,255).astype(dtype)
	return img

def visualize_heatmap(image,heatmap,plotTitle,cmap,savePath):
	"""FUNCTION::VISUALIZE_HEATMAP: Allows you to plot the heatmap generated for some methods.
		---
		Arguments:
		---
		>- image {np.array} -- Original Image data.
		>- heatmap {np.array} -- Heatmap obtained from a method.
		>- plotTitle {string} -- Title of the plot.
		>- cmap {string} -- Matplotlib color map.
		>- savePath {string} -- String specifying the save path of the plot.
		Returns:
		---
		>- {NONE} """
	figure = plt.figure(figsize=(6, 4))
	plt.subplot(121)
	plt.title('Raw Image')
	plt.axis('off')
	plt.imshow(image)
	plt.subplot(122)
	plt.title(plotTitle)
	plt.axis('off')
	plt.imshow(heatmap,interpolation='bilinear',cmap=cmap)
	figure.savefig(savePath,dpi=250)