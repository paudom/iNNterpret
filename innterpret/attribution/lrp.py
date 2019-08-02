from __future__ import absolute_import

# -- EXTERN IMPORTS -- #
from keras.models import Model
import keras.backend as K
import numpy as np

# -- IMPORT -- #
from .. import __verbose__ as vrb
from ..utils.tensor import model_remove_softmax, get_model_parameters, print_tensor_shape
from ..utils.bases.rules import ZPlus, ZAlpha, availableRules
from ..utils.data import load_image, reduce_channels, deprocess_image, visualize_heatmap
from ..utils.interfaces import Method
from ..utils.exceptions import OptionNotSupported

class LRPModel(Method):
	"""CLASS::LRPModel:
		---
		Description:
		---
		> Method (Layer-Wise Relevance Propagation), gets the attribution of each pixel.
		Arguments:
		---
		>- model {keras.Model} -- Model to analyze.
		Raises:
		---
		>- OptionNotSupported {Exception} -- When the rule option is not available.
		>- ValueError {Exception} -- If the value introduced for alpha is not correct.
		Link:
		---
		>- http://arxiv.org/abs/1706.07979."""
	def __init__(self,model):
		vrb.print_msg('self.__class__.__name__'+' Initialization...')
		vrb.print_msg('--------------------------')
		optionString = ''.join(option+':('+str(k)+') - ' for option,k in availableRules.items())
		self.ruleOption = int(input(vrb.set_msg(optionString[:-2])))
		if self.ruleOption not in list(availableRules.values()):
			raise OptionNotSupported('The option "'+str(self.ruleOption)+'" is not available.')
		if self.ruleOption == 1:
			self.alpha = int(input(vrb.set_msg('Select a Value for Alpha: ')))
			if self.alpha < 1:
				raise ValueError('Alpha value needs to be greater than 1.')
		reluModel = model_remove_softmax(model)
		_,_,activations,_ = get_model_parameters(model) 
		self.model = Model(inputs=reluModel.input, outputs=activations)
		self.numLayers = len(self.model.layers)
		vrb.print_msg('========== DONE ==========\n')

	def __define_rules(self,imgData,oneHot=False):
		"""METHOD::__DEFINE_RULES: goes through the model to save the specified rules.
			---
			Arguments:
			---
			>- imgData {np.array} -- Array representing the image data.
			>- oneHot {bool} -- Flag specifying to take the entire predictions or just the max value. (default:{False}).
			Returns:
			---
			>- {NONE}. """
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
		for k,currLayer in zip(range(self.numLayers-2,-1,-1),reversed(self.model.layers)):
			nextLayer = currLayer._inbound_nodes[0].inbound_layers[0]
			activation = self.outputR[k-1] if (k-1!=-1) else imgData
			if self.ruleOption == '0':
				layerRules.append(ZPlus(currLayer,activation))
			else:
				layerRules.append(ZAlpha(currLayer,activation,self.alpha))
			vrb.print_msg('<><><><><>')
			vrb.print_msg('Weights From: ['+currLayer.name+']')
			vrb.print_msg('Activations From: ['+nextLayer.name+']')
		self.rules = layerRules
		vrb.print_msg('========== DONE ==========\n')

	def __run_rules(self):
		"""METHOD::__RUN_RULES: computes the relevance tensor for all the model layers.
			---
			Returns:
			>- {NONE}."""
		vrb.print_msg('Run Rules...')
		vrb.print_msg('--------------------------')
		R = {}
		R[self.rules[0].name] = K.identity(self.outputR[-1])
		self.__print_rule_information(R[self.rules[0].name])
		for k,rule in enumerate(self.rules):
			if k != len(self.rules)-1:
				R[self.rules[k+1].name] = rule(R[rule.name],ignoreBias=False)
				self.__print_rule_information(R[self.rules[k+1].name])
			else:
				R['input'] = rule(R[rule.name],ignoreBias=False)
				self.__print_rule_information(R['input'])
		vrb.print_msg('========== DONE ==========\n')
		self.R = R

	def __print_rule_information(self,R):
		"""METHOD::__PRINT_RULE_INFORMATION: prints out the tensor shape of the rules runned.
			---
			Arguments:
			---
			>- R {tensor} -- Relevance tensor.
			Returns:
			---
			>- {NONE}."""
		if vrb.flag:
			shape = print_tensor_shape(R)
			vrb.print_msg('<><><><><>')
			vrb.print_msg('Rule R[input] Correctly runned.')
			vrb.print_msg('Tensor Output Shape: '+shape)

	def visualize(self,savePath,cmap='plasma',option='sum'):
		"""METHOD::VISUALIZE:
			---
			Arguments:
			---
			>- savePath {string} -- Path where the graph will be saved.
			>- cmap {string} -- Color map to represent the graph. (default:{'plasma'})
			>- option {string} -- How to represent the result. (default:{'sum'}).
			>>- 'sum'.
			>>- 'mean'.
			Returns:
			---
			>- {NONE}."""
		vrb.print_msg('Visualize '+self.__class__.__name__+' Result...')
		vrb.print_msg('--------------------------')
		oneDim = reduce_channels(self.relevance.copy(),option=option)
		heatMap = deprocess_image(oneDim.copy())
		visualize_heatmap(self.rawData,heatMap[0],self.__class__.__name__,cmap,savePath)
		vrb.print_msg('========== DONE ==========\n')

	def interpret(self,fileName,sess,layerName=None,oneHot=False):
		"""METHOD::INTERPRET:
			---
			Arguments:
			---
			>- fileName {string} -- The path of the image file.
			>- sess {tensorflow.Session} -- Session to compute all tensors at the same session.
			>- layerName {string} -- The layer selected to analyze. (default:{None})
			>- oneHot {bool} -- Determine if using only max activation or entire. (default:{False}).
			Returns:
			---
			>- {np.array} -- The resulting relevance array."""
		vrb.print_msg(self.__class__.__name__+' Analyzing')
		vrb.print_msg('--------------------------')
		self.rawData,imgData = load_image(fileName,preprocess=True)
		self.__define_rules(imgData,oneHot=oneHot)
		self.__run_rules()
		if layerName is None:
			self.relevance = sess.run(self.R['input'])
		else:
			self.relevance = sess.run(self.R[layerName])
		vrb.print_msg('========== DONE ==========\n')
		return self.relevance

	def __repr__(self):
		return super().__repr__()+'Layer-Wise Relevance Propagation (LRP)>'
