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
	""">> LOAD_VGG16: Load pretrained VGG16 model from keras."""
	if trained:
		return keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg16.VGG16(weights=None, include_top=True)

def load_vgg19(trained=True):
	""">> LOAD_VGG19: Load pretrained VGG19 model from keras."""
	if trained:
		return keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg19.VGG19(weights=None, include_top=True)

def load_model(h5file):
	""">> LOAD_MODEL: Loads model from a H5 file."""
	cwd = os.getcwd()
	try:
		model = keras.models.load_model(cwd+os.sep+h5file)
	except Exception:
		assert False
	else:
		return model
			#print_msg('Error while trying to load model from '+h5file+'.Try to execute '
			#'innterpret.utils.tensor_utils.fix_layer0()" If your model has an "InputLayer".',show=False,option='error')

def decode_predictions(predictions,top=5):
	""">> DECODE_PREDICTIONS: returns a list with the top predictions.
		The numbers specify the resulting neuron."""
	return predictions.argsort()[0][-top:][::-1]

def vgg_classes(idx=None):
	""">> VGG_CLASSES: returns a list with all the name classes.
		Their index indicate the number class for the VGG models."""
	pred = np.array([range(1000)[::-1]])
	decoded = keras.applications.vgg16.decode_predictions(pred,top=1000)
	result = [item[1] for item in decoded[0]]
	if idx is None:
		return result
	else:
		return result[idx]
 
def fix_layer0(h5file, inputShape, dtype):
	""">> FIX_LAYER0: Corrects an error when keras or tensorflow are loaded and have InputLayers."""
	with h5py.File(h5file,'r+') as file:
		modelConfig = json.loads(file.attrs['model_config'].decode('utf-8'))
		layer0 = modelConfig['config']['layers'][0]['config']
		layer0['batch_input_shape'] = inputShape
		layer0['dtype'] = dtype
		file.attrs['model_config'] = json.dumps(modelConfig).encode('utf-8')

def model_remove_softmax(model):
	""">> MODEL_REMOVE_SOFTMAX: Change the softmax activation of the last layer from a model."""
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
	""">> PRINT_TENSOR_SHAPE: Prints the shape of a tensor."""
	shape = '('
	for s in range(len(tensor.shape)):
		shape += str(tensor.shape[s].value)+','
	return shape[0:-1]+')'

def get_model_parameters(model):
	""">> GET_MODEL_PARAMETERS: Gets all the parameters of a model."""
	layerNames = []; layerOutputs = []; layerWeights = []; layerAct = []
	for layer in model.layers:
		layerNames.append(layer.name)
		layerOutputs.append(layer.output)
		layerAct.append(layer.activation)
		layerWeights.append(layer.get_weights)
	return layerNames,layerWeights,layerOutputs,layerAct



