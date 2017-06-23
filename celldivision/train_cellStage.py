from __future__ import print_function
from candidates import getStageLabel
import re
import cv2
import time
import keras
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

"""
def natural_key(string_):
    #See http://www.codinghorror.com/blog/archives/001018.html
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

batch_size = 1024
num_classes = 4
epochs = 100
data_augmentation = False

datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 100
numtrain = int(numvideos*0.7)
numtest = numvideos - numtrain

videos = os.listdir(datapath)
videos = videos[0:numvideos]

trainvideos = videos[0:numtrain]
testvideos = videos[numtrain:numtrain+numtest]

trainvideos = sorted(trainvideos,key=natural_key)
testvideos = sorted(testvideos,key=natural_key)

frame_array_train = []
label_array_train = []

for video in trainvideos: 
	print('Train: Video ' + video + ' from ' +str(len(trainvideos)))
	frames = os.listdir(os.path.join(datapath	,video))
	frames = sorted(frames,key=natural_key)
	pathL = os.path.join(datapath,video, '_trajectories.txt')
	with open(pathL) as f:
	        content = np.loadtxt(f)
	frames = frames[0:len(content)]
	labels = getStageLabel(os.path.join(datapath,video))
	for frame in frames:
		print('Train: Frame ' + frame + ' from ' +str(len(frames)))
		if os.path.join(datapath,video,frame).endswith('png'):
			im = cv2.imread(os.path.join(datapath,video,frame))
			im = cv2.resize(im,(180,180))
		frame_array_train.append(im)
	label_array_train.append(labels)

labels_train = []
for video_labels in label_array_train:
	labels_train = np.concatenate((labels_train,video_labels))

x_train = np.array(frame_array_train)
y_train = y_train = keras.utils.to_categorical(labels_train, num_classes)

frame_array_test = []
label_array_test = []

for video in testvideos: 
	print('Test: Video ' + video + ' from ' + str(len(testvideos)))
	frames = os.listdir(os.path.join(datapath	,video))
	frames = sorted(frames,key=natural_key)
	pathL = os.path.join(datapath,video, '_trajectories.txt')
	with open(pathL) as f:
	        content = np.loadtxt(f)
	frames = frames[0:len(content)]
	labels = getStageLabel(os.path.join(datapath,video))
	for frame in frames:
		print('Test: Frame ' + frame + ' from ' +str(len(frames)))
		if os.path.join(datapath,video,frame).endswith('png'):
			im = cv2.imread(os.path.join(datapath,video,frame))
			im = cv2.resize(im,(180,180))
		frame_array_test.append(im)
	label_array_test.append(labels)

labels_test = []
for video_labels in label_array_test:
	labels_test = np.concatenate((labels_test,video_labels))

x_test = np.array(frame_array_test)
y_test = keras.utils.to_categorical(labels_test, num_classes)
"""
da = np.load('datos_stage/StageData.npz')

x_train = da['x_train']
y_train = da['y_train']
x_test = da['x_test']
y_test = da['y_test']

###############--------------------------- Until Here Candidates Extraction -------------------------------###################################

targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models_stage/'
model_name = 'without0_antesdemonstruo_trial'

batch_size = 256
num_classes = 4
epochs = 100
data_augmentation = False

model = Sequential()

def step_decay(epoch):
  lrate_initial = 0.001
  drop = 0.1
  if epoch >= 25:
    lrate = lrate_initial * drop
  else:
    lrate = lrate_initial
  print('lrate',lrate)
  return lrate

model.add(Conv2D(16, (1, 1), padding='valid',
          input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(16, (3, 3),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Conv2D(32, (1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

"""
model.add(Conv2D(32, (1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
"""
model.add(Conv2D(64, (1, 1), padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3),padding='valid'))
model.add(Activation('relu'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(4, (1, 1),padding='valid'))
model.add(Activation('relu'))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_classes))
model.add(Activation('softmax'))

#opt =keras.optimizers.SGD(lr=0.00, momentum=0.9, decay=0.001)
#opt =keras.optimizers.Adagrad(lr=0.00, epsilon=1e-08, decay=0.001)
opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

class_weight = {0 : 1,
    1: 6.,
    2: 2.,
    3: 3.}

if not data_augmentation:
    print('Not using data augmentation.')
    print('Start Training')
    start = time.time()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=callbacks_list,class_weight = class_weight)
    end = time.time()
    elapsed_time = end-start
    print('Train time', elapsed_time)

else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
                                 featurewise_center=False, # set input mean to 0 over the dataset
                                 samplewise_center=False, # set each sample mean to 0
                                 featurewise_std_normalization=False, # divide inputs by std of the dataset
                                 samplewise_std_normalization=False, # divide each input by its std
                                 zca_whitening=False, # apply ZCA whitening
                                 rotation_range=45, # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=True, # randomly flip images
                                 vertical_flip=True) # randomly flip images
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

classes = model.predict(x_test, batch_size=64)

y_pred = []
for i in range(classes.shape[0]):
    y_pred.append(np.argmax(classes[i, :]))
y_pred = np.array((y_pred))

y_test_ = []
for i in range(y_test.shape[0]):
    y_test_.append(np.argmax(y_test[i, :]))
y_test_ = np.array((y_test_))

target_names = ['0', 'Stage1', 'Stage2', 'Stage3']
classificationReport = classification_report(y_test_, y_pred, target_names=target_names)
print(classificationReport)

metrics_train = model.evaluate(x_train,y_train,batch_size=batch_size)
metrics_test = model.evaluate(x_test,y_test,batch_size=batch_size)

#model.save(os.path.join(targetdir,model_name +'.h5'))
model.save(os.path.join(targetdir,model_name +'.h5'))
model.save_weights(os.path.join(targetdir,model_name +'_weights.h5'))
with open (os.path.join(targetdir,model_name + '.txt'),'w') as f:
  f.write(' NumTrainSamples: '  + str(len(x_train))
  	+ ' Class_weights: ' + str(class_weight)
    + ' NumTestSamples: ' +str(len(x_test))
    + ' NumVideos: ' + str(100) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   + '\n'
    + ' Batch_size: ' + str(batch_size) + '\n' 
    + ' Training Time: ' + str(elapsed_time) + '\n'
    + ' Data Augmentation: ' + str(data_augmentation)
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))
