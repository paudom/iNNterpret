# -- TENSOR UTILITIES -- #
from __future__ import absolute_import

# -- IMPORTS -- #
from .. import print_msg
import keras.backend as K
import numpy as np
import keras
import json 
import h5py 
import os 

# >> LOAD_VGG16: Load pretrained VGG16 model from keras.
def load_vgg16(trained=True):
	if trained:
		return keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg16.VGG16(weights=None, include_top=True)

# >> LOAD_VGG19: Load pretrained VGG19 model from keras.
def load_vgg19(trained=True):
	if trained:
		return keras.applications.vgg19.VGG19(weights='imagenet', include_top=True)
	else:
		return keras.applications.vgg19.VGG19(weights=None, include_top=True)

# >> LOAD_MODEL: Loads model from a H5 file.
def load_model(h5file):
	cwd = os.getcwd()
	try:
		model = keras.models.load_model(cwd+os.sep+h5file)
	except Exception as e:
		assert False, print_msg('Error while trying to load model from '+h5file+'.Try to execute '
			'innterpret.utils.tensor_utils.fix_layer0()" If your model has an "InputLayer".',show=False,option='error')

# >> DECODE_PREDICTIONS: returns a list with the top predictions. The numbers specify the resulting neuron.
def decode_predictions(predictions,top=5):
	return prediction.argsort()[0][-top:][::-1]

# >> VGG_CLASSES: returns a list with all the name classes. Their index indicate the number class for the VGG models.
def vgg_classes(idx=None):
	pred = np.array([range(1000)[::-1]])
	decoded = keras.applications.vgg16.decode_predictions(pred,top=1000)
	result = [item[1] for item in decoded[0]]
	if idx is None:
		return result
	else:
		return result[idx]

# >> FIX_LAYER0: Corrects an error when keras or tensorflow are loaded and have InputLayers. 
def fix_layer0(h5file, inputShape, dtype):
	with h5py.File(h5file,'r+') as file:
		modelConfig = json.loads(file.attrs['model_config'].decode('utf-8'))
		layer0 = modelConfig['config']['layers'][0]['config']
		layer0['batch_input_shape'] = inputShape
		layer0['dtype'] = dtype
		file.attrs['model_config'] = json.dumps(modelConfig).encode('utf-8')

# >> VGG_REMOVE_SOFTMAX: Change the softmax activation of the last layer from a VGG model.
def vgg_remove_softmax(model):
	assert model.layers[-1].activation.__name__ == keras.activations.softmax.__name__
	weights = model.layers[-1].get_weights()
	model.layers.pop()
	model.layers[-1].outbound_nodes = []
	model.outputs = [model.layers[-1].output]
	output = keras.layers.Dense(1000,activation='relu', name='predictions')(model.outputs[0])
	reluModel = keras.models.Model(inputs=model.inputs, outputs=[output])
	reluModel.layers[-1].set_weights(weights)
	return reluModel

# >> MODEL_REMOVE_SOFTMAX: Change the softmax activation of the last layer from a model.
def model_remove_sofmax(model):
	assert model.layers[-1].activation.__name__ == keras.activations.softmax.__name__
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

# >> PRINT_TENSOR_SHAPE: Prints the shape of a tensor.
def print_tensor_shape(tensor):
	shape = '('
	for s in range(len(tensor.shape)):
		shape += str(tensor.shape[s].value)+','
	return shape[0:-1]+')'

# >> GET_MODEL_PARAMETERS: Gets all the parameters of a model.
def get_model_parameters(model):
	layerNames = []; layerOutputs = []; layerWeights = []; layerAct = []
	for layer in model.layers:
		layerNames.append(layer.name)
		layerOutputs.append(layer.output)
		layerAct.append(layer.activation)
		layerWeights.append(layer.get_weights)
	return layerNames,layerWeights,layerOutputs,layerAct



