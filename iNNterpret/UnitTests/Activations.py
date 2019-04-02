# -- LOG -- #
import logging as log

log.basicConfig(filename='./Logs/visual_1.log',
				filemode='w',
				format='[%(asctime)s]:[%(name)s]:(%(levelname)s): %(message)s',
				level=log.INFO)

# -- IMPORTS -- #
try:
	from keras import models as kerasModels
	import matplotlib as mlp
	mlp.use('TkAgg')
	import matplotlib.pyplot as plt
	import json
	import h5py
	import numpy as np
	import pickle
except Exception as e:
	log.error('Failed -- IMPORT:\n',exc_info=True)
	raise SystemExit(0)

# -- FIX MODEL -- #
def fix_layer0(filename, batchShape, dtype):
	with h5py.File(filename, 'r+') as f:
		model_config = json.loads(f.attrs['model_config'].decode('utf-8'))
		layer0 = model_config['config']['layers'][0]['config']
		layer0['batch_input_shape'] = batchShape
		layer0['dtype'] = dtype
		f.attrs['model_config'] = json.dumps(model_config).encode('utf-8')

# -- LOAD -- #
try:
	model = kerasModels.load_model('./Models/vgg16_mnist.h5')
except Exception as e:
	log.info('Fixing Input Layer:\n')
	fix_layer0('./Models/vgg16_mnist.h5',[None,32,32,3],'float32')
	model = kerasModels.load_model('./Models/vgg16_mnist.h5')
with open('./Data/Mnist_data.pkl','rb') as file:
	Xtest,Ytest = pickle.load(file)
log.info('LOAD: SUCCESS')

# -- VARIABLES FOR VISUALIZATION -- #
layerNames = []
layerOutputs = []
for layer in model.layers:
    if layer.__class__.__name__ in ['Conv2D','MaxPooling2D']:
        layerOutputs.append(layer.output)
        layerNames.append(layer.name)
visual = kerasModels.Model(inputs=model.input,outputs=layerOutputs)

# -- VISUALIZATION OF ALL ACTIVATIONS GIVEN A IMAGE -- #
selected = int(input('Select from (0-{0}): '.format(Xtest.shape[0]-1)))
imageInput = np.reshape(Xtest[selected],(1,32,32,3))
activations = visual.predict(imageInput)
os.mkdir('./Data/VGG16')
cols = 32
for n,act in zip(range(len(activations)),activations):
    if layerTypes[n] in ['Conv2D','MaxPooling2D']:
        numFilters = act.shape[-1]; size = act.shape[1]
        rows = numFilters // cols
        fig = plt.figure(figsize=(cols,rows))
        for k in range(0,cols*rows):
            image = act[0,:,:,k]
            image = (image-np.min(image))/(np.max(image)-np.min(image))
            fig.add_subplot(rows,cols,k+1)
            plt.imshow(image,cmap=plt.cm.gray)
            plt.xticks([]); plt.yticks([])
        fileName = './Data/VGG16/'+layerNames[n]+'.png'
        fig.savefig(fileName)