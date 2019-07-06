from __future__ import absolute_import

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.tensor import model_remove_softmax, get_model_parameters, print_tensor_shape
from ..utils.bases.rules import ZPlus, ZAlpha
from ..utils.data import load_image, reduce_channels, deprocess_image, visualize_heatmap
from keras.models import Model
import keras.backend as K
import numpy as np

class LRPModel():
	""">> CLASS:LRPMODEL: Method (Layer-Wise Relevance Propagation): http://arxiv.org/abs/1706.07979."""
	def __init__(self,model):
		vrb.print_msg('self.__class__.__name__'+' Initialization...')
		vrb.print_msg('--------------------------')
		reluModel = model_remove_softmax(model)
		_,_,activations,_ = get_model_parameters(model) 
		self.model = Model(inputs=reluModel.input, outputs=activations)
		if vrb.flag:
			self.model.summary()
		self.optionRule = input(vrb.set_msg('(ZPlus)-(ZAlpha): '))
		if self.optionRule == 'ZAlpha':
			self.alpha = int(input(vrb.set_msg('Select a Value for Alpha: ')))
			#assert self.alpha >= 1
		self.numLayers = len(self.model.layers)
		vrb.print_msg('========== DONE ==========\n')

	def define_rules(self,imgData,oneHot=False):
		""">> DEFINE_RULES: goes through the model to save the specified rules."""
		vrb.print_msg('Define Rules...')
		vrb.print_msg('--------------------------')       
		layerRules = []; self.outputR = self.model.predict(imgData)
		# Leave only the max activated class, and the others to 0
		if oneHot:
			clsNeuron = np.argmax(self.outputR[-1])
			maxAct = self.outputR[-1][:,clsNeuron]
			self.outputR[-1] = np.zeros(self.outputR[-1].shape, dtype='float32')
			self.outputR[-1][:,clsNeuron] = maxAct
		# For each layer backwards, define the Rules.
		for currLayer,k in zip(reversed(self.model.layers),range(self.numLayers-2,-1,-1)):
			nextLayer = currLayer._inbound_nodes[0].inbound_layers[0]
			activation = self.outputR[k-1] if (k-1!=-1) else imgData
			if self.optionRule == 'ZPlus':
				layerRules.append(ZPlus(currLayer,activation))
			elif self.optionRule == 'ZAlpha':
				layerRules.append(ZAlpha(currLayer,activation,self.alpha))
			#else:
				#assert False, 'This Rule Option is not supported'
			vrb.print_msg('<><><><><>')
			vrb.print_msg('Weights From: ['+currLayer.name+']')
			vrb.print_msg('Activations From: ['+nextLayer.name+']')
		self.rules = layerRules
		vrb.print_msg('========== DONE ==========\n')

	def run_rules(self):
		""">> RUN_RULES: computes the relevance tensor for all the model layers."""
		vrb.print_msg('Run Rules...')
		vrb.print_msg('--------------------------')
		R = {}
		R[self.rules[0].name] = K.identity(self.outputR[-1])
		self.print_rule_information(R[self.rules[0].name])
		for k in range(len(self.rules)):
			if k != len(self.rules)-1:
				R[self.rules[k+1].name] = self.rules[k].run(R[self.rules[k].name],ignoreBias=False)
				self.print_rule_information(R[self.rules[k+1].name])
			else:
				R['input'] = self.rules[k].run(R[self.rules[k].name],ignoreBias=False)
				self.print_rule_information(R['input'])
		vrb.print_msg('========== DONE ==========\n')
		self.R = R

	def print_rule_information(self,R):
		""">> PRINT_RULE_INFORMATION: prints out the tensor shape of the rules runned."""
		if vrb.flag:
			shape = print_tensor_shape(R)
			vrb.print_msg('<><><><><>')
			vrb.print_msg('Rule R[input] Correctly runned.')
			vrb.print_msg('Tensor Output Shape: '+shape)

	def visualize(self,savePath,cmap='plasma',option='sum'):
		""">> VISUALIZE: returns a graph with the results."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		oneDim = reduce_channels(self.relevance.copy(),option=option)
		heatMap = deprocess_image(oneDim.copy())
		visualize_heatmap(self.rawData,heatMap[0],self.__class__.__name__,cmap,savePath)
		vrb.print_msg('========== DONE ==========\n')

	def execute(self,fileName,sess,layerName=None,oneHot=False):
		""">> EXECUTE: returns the result of the method."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData = load_image(fileName,preprocess=False)
		imgData = load_image(fileName)
		self.define_rules(imgData,oneHot=oneHot)
		self.run_rules()
		if layerName is None:
			self.relevance = sess.run(self.R['input'])
		else:
			self.relevance = sess.run(self.R[layerName])
		vrb.print_msg('========== DONE ==========\n')
		return self.relevance
