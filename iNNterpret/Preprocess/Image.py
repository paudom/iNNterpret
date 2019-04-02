# -- PREPROCESS IMAGE UTILS -- #
	
## -- IMPORTS -- ##
from PIL import Image as pilImage
import numpy as np

# -- IMG_TO_VECTOR -- #: Converts one image into one dimensional vector. 
def img_to_vector(img,linear=True):
	img = np.asarray(img,dtype='float32')
	H,W,Z = img.shape
	if linear:
		R = img[:,:,0].reshape(H*W,1)
		G = img[:,:,1].reshape(H*W,1) 
		B = img[:,:,2].reshape(H*W,1)
		vector = np.concatenate([R,G,B],axis=0)
	else:
		vector = img.reshape(H*W*Z,1)
	assert vector.shape[0]==H*W*Z, f'Vector must be of size ({H*W*Z},{1})'
	return vector,H,W

# -- VECTOR_TO_IMG -- #: Converts one dimensional vector into an image.
def vector_to_img(array,height,width):
	return array.reshape(height,width,3)

# -- NORM_IMG -- #: Normalizes image data.
def norm_img(vector,axis=-1,order=2):
	norm = np.atleast_1d(np.linalg.norm(vector,order,axis))
	norm[norm == 0] = 1
	return vector/np.expand_dims(norm,axis)

# -- SCALE_IMG -- #: Scales the image to [0,255] domain.
def scale_img(array):
	array = array + max(-np.min(array), 0); arr_max = np.max(array)
	if arr_max != 0:
		array /= arr_max
	array *= 255
	return array

# -- COLOR_CONVERT -- #: Converts an image into  another with the specified color space.
def color_convert(img,colorSpace):
	if colorSpace == 'rgb' and img.mode != 'RGB':
		return img.convert('RGB')
	elif colorSpace == 'rgba' and img.mode != 'RGBA':
		return img.convert('RGBA')
	elif colorSpace == 'grayscale' and img.mode != 'L':
		return img.convert('L')
	elif colorSpace == 'yuv' and img.mode != 'YCbCr':
		return img.convert('YCbCr')
	elif colorSpace == 'hsv' and img.mode != 'HSV':
		return img.convert('HSV')
	else:
		raise ValueError('colorSpace is not valid.')  
