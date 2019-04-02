# -- LOG -- #
import logging as log

log.basicConfig(filename='./Logs/vgg16_mnist.log',
				filemode='w',
				format='[%(asctime)s]:[%(name)s]:(%(levelname)s): %(message)s',
				level=log.INFO)


# -- IMPORT LIBRARIES -- #
try:
	from keras.applications.vgg16 import VGG16
	from keras.datasets import mnist
	from keras.preprocessing import image as kerasImage
	from keras.callbacks import EarlyStopping, ModelCheckpoint
	from keras.models import Sequential
	from keras.layers import Flatten, Dense, Dropout, Activation
	from keras.optimizers import RMSprop, Adam
	from PIL import Image as pilImage
	import keras.utils as kerasUtils
	import numpy as np
	import pickle
except Exception as e:
	log.error('Failed -- IMPORT:\n',exc_info=True)
	raise SystemExit(0)

# -- DEFINE VARIABLES -- #
inputShape = (48,48,3)
imgSize = (48,48)

# -- DEFINE FUNCTIONS -- #
def split_validation(Xtrain,Ytrain,splitRatio):
	total = Xtrain.shape[0]
	cutTrain = int(round(total*(1-splitRatio), 0))
	trainSet = (Xtrain[0:cutTrain,:,:,:],Ytrain[0:cutTrain,:])
	validSet = (Xtrain[cutTrain:,:,:,:],Ytrain[cutTrain:,:])
	return trainSet,validSet

# -- LOAD DATA -- #
(Xtrain,Ytrain),(Xtest,Ytest) = mnist.load_data()
Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)
Ytrain = np.expand_dims(Ytrain,-1)
log.info('LOAD DATA: SUCCESS.')

# -- REDUCE DATA -- #
(Xtrain,Ytrain),_ = split_validation(Xtrain,Ytrain,1/3)
log.info('REDUCE DATA: SUCCESS.')

# -- PREPARE DATA -- #
try:
	Xtrain = np.asarray([kerasUtils.normalize(
			kerasImage.img_to_array(
			kerasImage.array_to_img(im,scale=False).convert('RGB').resize(imgSize)),axis=0) for im in Xtrain])
	Xtest = np.asarray([kerasUtils.normalize(
			kerasImage.img_to_array(
			kerasImage.array_to_img(im,scale=False).convert('RGB').resize(imgSize)),axis=0) for im in Xtest])
	Ytrain = kerasUtils.to_categorical(Ytrain)
	Ytest = kerasUtils.to_categorical(Ytest)
	log.info('PREPARE DATA: SUCCESS.')
except Exception as e:
	log.error('Failed -- PREPARE DATA:\n',exc_info=True)
	raise SystemExit(0)

# -- SAVE DATA -- #
with open('./Data/Mnist_data.pkl','wb') as file:
	pickle.dump((Xtest,Ytest),file)
log.info('SAVE DATA: SUCCESS.')

# -- DEFINE MODEL -- #
try:
	vgg16Model = VGG16(weights='imagenet',include_top=False,input_shape=inputShape)
	model = Sequential()
	for layer in vgg16Model.layers:
		layer.trainable = False
		model.add(layer)
	transferLayers = [Flatten(),
					 Dense(512,activation='relu'),
					 Dense(512,activation='relu'),
					 Dense(10,activation='softmax')]
	for layer in transferLayers:
		model.add(layer)
	for layer in model.layers[:-len(transferLayers)]:
		assert layer.trainable == False
	log.info('DEFINE MODEL: SUCCESS')
except Exception as e:
	log.error('Failed -- DEFINE MODEL:\n',exc_info=True)
	raise SystemExit(0)

# -- DEFINE VARIABLES FOR MODEL TRAINING -- #
batchSize = 64
fitEpochs = 10
(Xtrain,Ytrain),(Xvalid,Yvalid) = split_validation(Xtrain,Ytrain,1/4)

# -- COMPILE MODEL -- #
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

earlyStop = EarlyStopping(monitor='val_loss',  
						  min_delta=0.01,
						  patience=2,
						  verbose=1,
						  mode='auto')

checkpointer = ModelCheckpoint(filepath="./Data/vgg16_mnist_weigths.hdf5", 
							   monitor = 'val_acc',
							   verbose=1, 
							   save_best_only=True)
log.info('COMPILE MODEL: SUCCESS')

# -- TRAIN MODEL -- #
try:
	history = model.fit(Xtrain,Ytrain,
				epochs=fitEpochs,
				batch_size=batchSize,
				validation_data=(Xvalid,Yvalid),
				callbacks=[earlyStop,checkpointer])
	log.info('TRAIN MODEL: SUCCESS')
except Exception as e:
	log.error('FAILED -- TRAIN MODEL:\n',exc_info=True)
	raise SystemExit(0)


# -- LOAD MODEL WEIGHTS -- #
model.load_weights('./Data/vgg16_mnist_weigths.hdf5')
log.info('LOAD MODEL WEIGHTS: SUCCESS.')
for layer in model.layers:
		layer.trainable = False

# -- VALIDATE TRAINING -- #
loss,accuracy = model.evaluate(Xvalid,
							   Yvalid,
							   batch_size=batchSize)
log.info('---- VALIDATION ----')
log.info('Loss: {:.2%}'.format(loss))
log.info('Accuracy: {:.2%}'.format(accuracy))

# -- EVALUATE MODEL -- #
loss,accuracy = model.evaluate(Xtest,
							   Ytest,
							   batch_size=batchSize)
log.info('---- TEST ----')
log.info('Loss: {:.2%}'.format(loss))
log.info('Accuracy: {:.2%}'.format(accuracy))

# -- SAVE MODEL -- #
model.save('./Models/vgg16_mnist.h5')
log.info('SAVE MODEL: SUCCESS')
log.info('DONE')

