from __future__ import print_function
import candidates as ca
import keras
from keras.callbacks import LearningRateScheduler
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import random
from sklearn.metrics import classification_report

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

batch_size = 2048
num_classes = 3
epochs = 100

datapath = '/home/jcleon/MouEmbTrkDtb/'
numvideos = 30
numtrain = int(numvideos * 0.7)
numtest = numvideos - numtrain

videos = os.listdir(datapath)

trainvideos = videos[:numtrain]
testvideos = videos[numtrain + 1:]

voxelSize = 13
step = 20
timeSize = 1
tol = 0

voxel_array_train = []
labels_train = []

for i in range(0, numtrain):
    print('Video', trainvideos[i])
    videoCube_train = ca.loadVideoCube(os.path.join(datapath, trainvideos[i]))
    for x in range(0, videoCube_train.shape[0]-voxelSize, step):
        print('train_data', x)
        for y in range(0, videoCube_train.shape[1]-voxelSize, step):
            for z in range(0, videoCube_train.shape[2]-timeSize, step):
                #voxelDescriptor = getSTIPDescriptor(aVoxel)
                voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath, trainvideos[i]))
              
                if voxelLabel == 0:
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.8:
                        continue
              
                aVoxel = ca.getVoxelFromVideoCube(videoCube_train, x, y, z, voxelSize, timeSize)
                voxel_array_train.append(aVoxel)
                labels_train.append(voxelLabel)
                #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

x_train = np.array((voxel_array_train))
y_train = np.array((labels_train))
y_train = keras.utils.to_categorical(labels_train, num_classes)##ii


voxel_array_test = []
labels_test = []

for i in range(0, numtest):
    print('Video', trainvideos[i])
    videoCube_test = ca.loadVideoCube(os.path.join(datapath, testvideos[i]))
    for x in range(0, videoCube_test.shape[0]-voxelSize, step):
        print('test_data', x)
        for y in range(0, videoCube_test.shape[1]-voxelSize, step):
            for z in range(0, videoCube_test.shape[2]-timeSize, step):         
                #voxelDescriptor = getSTIPDescriptor(aVoxel)
                voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath, testvideos[i]))
                if voxelLabel == 0:
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.8:
                        continue
              
                aVoxel = ca.getVoxelFromVideoCube(videoCube_test, x, y, z, voxelSize, timeSize)
                voxel_array_test.append(aVoxel)
                labels_test.append(voxelLabel)
                #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

x_test = np.array((voxel_array_test))
y_test = np.array((labels_test))
y_test = keras.utils.to_categorical(labels_test, num_classes)


model = Sequential()

model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3), padding='same'))
model.add(Activation('relu'))


model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

classes = model.predict(x_test, batch_size=2048)
y_pred = []

for i in range(classes.shape[0]):
    y_pred.append(np.argmax(classes[i,:]))
y_pred = np.array((y_pred))
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labels_test, y_pred, target_names=target_names)
print(classificationReport)



