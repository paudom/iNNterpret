# -- TENSOR UTILITIES -- #
from __future__ import absolute_import

# -- IMPORTS -- #
import keras.backend as K
import numpy as np
import keras
import json 
import h5py 
import os 

def load_vgg16(trained=True):
	"""FUNCTION::LOAD_VGG16: Load pretrained VGG16 model from keras.
		---
		Arguments:
		---
		>- trained {bool} -- A flag indicating if the model loaded will be trained. (default:{True}).
		Returns:
		---
		>- {keras.Model} -- The VGG16 Model."""
	if trained:
		return keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg16.VGG16(weights=None, include_top=True)

def load_vgg19(trained=True):
	"""FUNCTION::LOAD_VGG19: Load pretrained VGG19 model from keras.
		---
		Arguments:
		---
		>- trained {bool} -- A flag indicating if the model loaded will be trained. (default:{True}).
		Returns:
		---
		>- {keras.Model} -- The VGG19 Model."""
	if trained:
		return keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg19.VGG19(weights=None, include_top=True)

def load_model(h5file):
	"""FUNCTION::LOAD_MODEL: Loads model from a H5 file.
		---
		Arguments:
		---
		>- h5file {string} -- Path to the h5 file containing the model.
		Returns:
		---
		>- {model} -- The model."""
	cwd = os.getcwd()
	try:
		model = keras.models.load_model(cwd+os.sep+h5file)
	except Exception:
		pass
	#	assert False
	else:
		return model
			#print_msg('Error while trying to load model from '+h5file+'.Try to execute '
			#'innterpret.utils.tensor_utils.fix_layer0()" If your model has an "InputLayer".',show=False,option='error')

def decode_predictions(predictions,top=5):
	"""FUNCTION::DECODE_PREDICTIONS: returns a list with the top predictions.
		---
		Arguments:
		---
		>- predictions {np.array} -- Array containing the predictions of the model.
		>- top {int} -- Number indicating the number of classes to show. (default:{5}).
		Returns:
		---
		>- {np.array} -- An array containing the classes with the higher probability."""
	return predictions.argsort()[0][-top:][::-1]

def vgg_classes(idx=None):
	"""FUNCTION::VGG_CLASSES: returns a list with all the name classes for the VGG16 classes.
		---
		Arguments:
		---
		>- idx {int} -- Indicates the name of the selected class. (default:{None})
		Returns:
		---
		>- {list[string]} -- Containing the name/s of the classes for the VGG16."""
	pred = np.array([range(1000)[::-1]])
	decoded = keras.applications.vgg16.decode_predictions(pred,top=1000)
	result = [item[1] for item in decoded[0]]
	if idx is None:
		return result
	else:
		return result[idx]
 
def fix_layer0(h5file, inputShape, dtype='float32'):
	"""FUNCTION::FIX_LAYER0: Corrects an error when keras or tensorflow MODELS are loaded and have InputLayers.
		---
		Arguments:
		---
		>- h5file {string} -- Path of the h5 file.
		>- inputShape {list[int]} -- Specify the input shape of the model input.
		>- dtype {string} -- Specify the type of the data. (default:{'float32'})
		Returns:
		---
		>- {NONE}"""
	with h5py.File(h5file,'r+') as file:
		modelConfig = json.loads(file.attrs['model_config'].decode('utf-8'))
		layer0 = modelConfig['config']['layers'][0]['config']
		layer0['batch_input_shape'] = inputShape
		layer0['dtype'] = dtype
		file.attrs['model_config'] = json.dumps(modelConfig).encode('utf-8')

def model_remove_softmax(model):
	"""FUNCTION::MODEL_REMOVE_SOFTMAX: Change the softmax activation of the last layer from a model.
		---
		Arguments:
		---
		>- model {keras.Model} -- A model you want to modify.
		Returns:
		---
		>- {keras.Model} -- Returns a model without the softmax activation at the end."""
	#assert model.layers[-1].activation.__name__ == keras.activations.softmax.__name__
	outShape = model.outputs[0].shape[-1]
	outName = model.layers[-1].name
	weights = model.layers[-1].get_weights()
	model.layers.pop()
	model.layers[-1].outbound_nodes = []
	model.outputs = [model.layers[-1].output]
	output = keras.layers.Dense(outShape,activation='relu', name=outName)(model.outputs[0])
	reluModel = keras.models.Model(inputs=model.inputs, outputs=[output])
	reluModel.layers[-1].set_weights(weights)
	return reluModel

def print_tensor_shape(tensor):
	"""FUNCTION::PRINT_TENSOR_SHAPE: Prints the shape of a tensor.
		---
		Arguments:
		---
		>- tensor {tensor} -- The tensor to analyze.
		Returns:
		>- {string} -- With the shape of the tensor."""
	shape = '('
	for s in range(len(tensor.shape)):
		shape += str(tensor.shape[s].value)+','
	return shape[0:-1]+')'

def get_model_parameters(model):
	"""FUNCTION::GET_MODEL_PARAMETERS: Gets all the parameters of a model.
		---
		Arguments:
		---
		>- model {keras.Model} -- The model to analyze.
		Returns:
		---
		>- {list[string]},{list[np.array]},{list[tensor]},{list[tensor]} -- layer Names, Weights, Outputs and Activations."""
	layerNames = []; layerOutputs = []; layerWeights = []; layerAct = []
	for layer in model.layers:
		layerNames.append(layer.name)
		layerOutputs.append(layer.output)
		layerAct.append(layer.activation)
		layerWeights.append(layer.get_weights)
	return layerNames,layerWeights,layerOutputs,layerAct



