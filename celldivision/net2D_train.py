from __future__ import print_function
import candidates as ca
from matplotlib import pyplot as plt
import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import os
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


with tf.device('/gpu:0'):
	batch_size = 32
	num_classes = 3
	epochs = 10
	data_augmentation = False

	datasetRoot = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb'
	trainFrames = os.path.join(datasetRoot, 'E01')
	testFrames = os.path.join(datasetRoot, 'E17')
	videoCube_train = ca.loadVideoCube(trainFrames)  
	videoCube_test = ca.loadVideoCube(testFrames) 

	voxelSize = 11
	step = 10
	timeSize = 1

	voxel_array_train = []
	labels_train = []

	for x in range(0, videoCube_train.shape[0]-voxelSize, step):
	    print('train_data', x)
	    
	    for y in range(0, videoCube_train.shape[1]-voxelSize, step):
	        for z in range(0, videoCube_train.shape[2]-timeSize, step):

	            aVoxel = ca.getVoxelFromVideoCube(videoCube_train, x, y, z, voxelSize, timeSize)
	            #voxelDescriptor = getSTIPDescriptor(aVoxel)
	            voxelLabel = ca.getCubeLabel(x, y, z, 5, trainFrames)
	            voxel_array_train.append(aVoxel)
	            labels_train.append(voxelLabel)
	            #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

	x_train = np.array((voxel_array_train))
	y_train = np.array((labels_train))
	y_train = keras.utils.to_categorical(labels_train, num_classes)


	voxel_array_test = []
	labels_test = []

	for x in range(0, videoCube_test.shape[0]-voxelSize, step):
	    print('test_data', x)
	    
	    for y in range(0, videoCube_test.shape[1]-voxelSize, step):
	        for z in range(0, videoCube_test.shape[2]-timeSize, step):

	            aVoxel = ca.getVoxelFromVideoCube(videoCube_test, x, y, z, voxelSize, timeSize)
	            #voxelDescriptor = getSTIPDescriptor(aVoxel)
	            voxelLabel = ca.getCubeLabel(x, y, z, 5, testFrames)
	            voxel_array_test.append(aVoxel)
	            labels_test.append(voxelLabel)
	            #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

	x_test = np.array((voxel_array_test))
	y_test = np.array((labels_test))
	y_test = keras.utils.to_categorical(labels_test, num_classes)


	model = Sequential()


	model.add(Conv2D(32, (3, 3), padding='same',
	                 input_shape=x_train.shape[1:]))
	model.add(Activation('relu'))
	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))

	opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

	model.compile(loss='categorical_crossentropy',
	              optimizer=opt,
	metrics=['accuracy'])

	if not data_augmentation:
	    print('Not using data augmentation.')
	    model.fit(x_train, y_train,
	              batch_size=batch_size,
	              epochs=epochs,
	              validation_data=(x_test, y_test),
	              shuffle=True)
	else:
	    print('Using real-time data augmentation.')
	    # This will do preprocessing and realtime data augmentation:
	    datagen = ImageDataGenerator(
	        featurewise_center=False,  # set input mean to 0 over the dataset
	        samplewise_center=False,  # set each sample mean to 0
	        featurewise_std_normalization=False,  # divide inputs by std of the dataset
	        samplewise_std_normalization=False,  # divide each input by its std
	        zca_whitening=False,  # apply ZCA whitening
	        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
	        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
	        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
	        horizontal_flip=True,  # randomly flip images
			vertical_flip=False) # randomly flip images
	    datagen.fit(x_train)
	    model.fit_generator(datagen.flow(x_train, y_train,
	                                     batch_size=batch_size),
	                        steps_per_epoch=x_train.shape[0] // batch_size,
	                        epochs=epochs,
							validation_data=(x_test, y_test))

	classes = model.predict(x_test, batch_size=128)
	y_pred = []
	for i in range(classes.shape[0]):
		y_pred.append(np.argmax(classes[i,:]))
	y_pred = np.array((y_pred))
	target_names = ['Background', 'Cell', 'Boundary']
	print(classification_report(labels_test, y_pred, target_names=target_names))
