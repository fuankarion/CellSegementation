from __future__ import print_function
import candidates as ca
import keras
from keras.layers import Activation
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten, add
from keras.layers import MaxPooling3D, Input, AveragePooling2D, Concatenate, AveragePooling3D
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
#from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import os
from random import randint

os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def load_MaskCandidates(NumVideos=30,voxel_Size=13,step=13,timeSize=1,timeStep=2,tol=0):
  datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
  numvideos = NumVideos
  numtrain = int(numvideos*0.7)
  numtest = numvideos - numtrain

  videos = os.listdir(datapath)
  videos = videos[0:numvideos]

  trainvideos = videos[0:numtrain]
  testvideos = videos[numtrain:numtrain+numtest]

  voxelSize = voxel_Size
  step = step
  timeSize = timeSize
  timeStep = timeStep
  tol = tol

  voxel_array_train = []
  labels_train = []

  for i in range(0,numtrain):
    print('Video',trainvideos[i])
    videoCube_train = ca.loadVideoCube(os.path.join(datapath,trainvideos[i]))
    for x in range(int(voxelSize/2), videoCube_train.shape[0]-int(voxelSize/2), randint(int(step/2),step)):
        print('train_data', x)
        for y in range(int(voxelSize/2), videoCube_train.shape[1]-int(voxelSize/2), randint(int(step/2),step)):
            for z in range(timeSize, videoCube_train.shape[3]-timeSize,timeStep): #randint(int(timeStep/2), timeStep)):
                #voxelDescriptor = getSTIPDescriptor(aVoxel)
                voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,trainvideos[i]))
                
                if voxelLabel == 0:
                       ignoreFlag = random.uniform(0.0, 1.0)
                       if ignoreFlag <= 0.8:
                           continue
                
                aVoxel = ca.getVoxelFromVideoCube(videoCube_train, x, y, z, voxelSize, timeSize)
                voxel_array_train.append(aVoxel)
                labels_train.append(voxelLabel)
                #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

  x_train = np.array((voxel_array_train))
  #x_train = np.squeeze(x_train)
  #x_train = np.expand_dims(x_train, axis=5)
  y_train = np.array((labels_train))
  y_train = keras.utils.to_categorical(labels_train, num_classes)


  voxel_array_test = []
  labels_test = []

  for i in range(0,numtest):
    print('Video',trainvideos[i])
    videoCube_test = ca.loadVideoCube(os.path.join(datapath,testvideos[i]))
    for x in range(int(voxelSize/2), videoCube_test.shape[0]-int(voxelSize/2), randint(int(step/2),step)):
        print('test_data', x)
        for y in range(int(voxelSize/2), videoCube_test.shape[1]-int(voxelSize/2), randint(int(step/2),step)):
            for z in range(timeSize, videoCube_test.shape[3]-timeSize, timeStep):#randint(int(timeStep/2), timeStep)):
                #voxelDescriptor = getSTIPDescriptor(aVoxel)
                voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,testvideos[i]))
                if voxelLabel == 0:
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.8:
                        continue
                
                aVoxel = ca.getVoxelFromVideoCube(videoCube_test, x, y, z, voxelSize, timeSize)
                voxel_array_test.append(aVoxel)
                labels_test.append(voxelLabel)
                #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

  x_test = np.array((voxel_array_test))
  #x_test = np.squeeze(x_test)
  #x_test = np.expand_dims(x_test, axis=5)
  y_test = np.array((labels_test))
  y_test = keras.utils.to_categorical(labels_test, num_classes)

  return x_train,y_train,x_test,y_test
"""
datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 100
numtrain = int(numvideos*0.7)
numtest = numvideos - numtrain

videos = os.listdir(datapath)
videos = videos[0:numvideos]

trainvideos = videos[0:numtrain]
testvideos = videos[numtrain:numtrain+numtest]

voxelSize = 13
step = 12
timeSize = 2
timeStep = 1
tol = 0

voxel_array_train = []
labels_train = []

for i in range(0,numtrain):
  print('Video',trainvideos[i])
  videoCube_train = ca.loadVideoCube(os.path.join(datapath,trainvideos[i]))
  for x in range(int(voxelSize/2), videoCube_train.shape[0]-int(voxelSize/2), randint(step/2,step)):
      print('train_data', x)
      for y in range(int(voxelSize/2), videoCube_train.shape[1]-int(voxelSize/2), randint(step/2,step)):
          for z in range(int(timeSize/2), videoCube_train.shape[3]-int(timeSize/2), randint(timeStep/2, timeStep)):
              #voxelDescriptor = getSTIPDescriptor(aVoxel)
              voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,trainvideos[i]))
              
              if voxelLabel == 0:
                     ignoreFlag = random.uniform(0.0, 1.0)
                     if ignoreFlag <= 0.8:
                         continue
              
              aVoxel = ca.getVoxelFromVideoCube(videoCube_train, x, y, z, voxelSize, timeSize)
              voxel_array_train.append(aVoxel)
              labels_train.append(voxelLabel)
              #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

x_train = np.array((voxel_array_train))
#x_train = np.expand_dims(x_train, axis=5)
y_train = np.array((labels_train))
y_train = keras.utils.to_categorical(labels_train, num_classes)


voxel_array_test = []
labels_test = []

for i in range(0,numtest):
  print('Video',trainvideos[i])
  videoCube_test = ca.loadVideoCube(os.path.join(datapath,testvideos[i]))
  for x in range(int(voxelSize/2), videoCube_test.shape[0]-int(voxelSize/2), randint(step/2,step)):
      print('test_data', x)
      for y in range(int(voxelSize/2), videoCube_test.shape[1]-int(voxelSize/2), randint(step/2,step)):
          for z in range(int(timeSize/2), videoCube_test.shape[3]-int(timeSize/2), randint(timeStep/2, timeStep)):
              #voxelDescriptor = getSTIPDescriptor(aVoxel)
              voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,testvideos[i]))
              if voxelLabel == 0:
                  ignoreFlag = random.uniform(0.0, 1.0)
                  if ignoreFlag <= 0.8:
                      continue
              
              aVoxel = ca.getVoxelFromVideoCube(videoCube_test, x, y, z, voxelSize, timeSize)
              voxel_array_test.append(aVoxel)
              labels_test.append(voxelLabel)
              #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

x_test = np.array((voxel_array_test))
#x_test = np.expand_dims(x_test, axis=5)
y_test = np.array((labels_test))
y_test = keras.utils.to_categorical(labels_test, num_classes)
"""

def SqueezeNet(nb_classes, inputs=(128,128,3)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """

    input_img = Input(shape=inputs)
    conv1 = Conv3D(
        32, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2,2,1), padding='same', name='conv1')(input_img)
    maxpool1 = MaxPooling3D(
        pool_size=(3,3,1), strides=(2,2,1), name='maxpool1')(conv1)

    fire2_squeeze = Conv3D(
        8, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Conv3D(
        16, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Conv3D(
        16, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])

    fire3_squeeze = Conv3D(
        8, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Conv3D(
        16, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Conv3D(
        16, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])

    residual32 = add([merge2, merge3])

    fire4_squeeze = Conv3D(
        16, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze')(residual32)
    fire4_expand1 = Conv3D(
        32, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Conv3D(
        32, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])
    #maxpool4 = MaxPooling3D(
    #    pool_size=(3,3,1), strides=(2,2,2), name='maxpool4')(merge4)

    fire5_squeeze = Conv3D(
        16, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze')(merge4)
    fire5_expand1 = Conv3D(
        32, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Conv3D(
        32, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])

    residual45 = add([merge4, merge5])

    fire6_squeeze = Conv3D(
        24, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Conv3D(
        64, (1,1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Conv3D(
        64, (3,3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])

    fire6_dropout = Dropout(0.5, name='fire6_dropout')(merge6)
    conv10 = Conv3D(
        nb_classes, (1,1,1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10')(fire6_dropout)

    # The size should match the output of conv10
    avgpool10 = AveragePooling3D(
        (2, 2, 3), name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    softmax = Activation("softmax", name='softmax')(flatten)

    return Model(inputs=input_img, outputs=softmax)


targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models_correct/3d/timeExploration/'
batch_size = 1024
num_classes = 2
epochs = 50
data_augmentation = False

#vSizes = np.array((11,13))
#tsizes = np.array((3,4,5,2))
numvideos = 100
vSize = 11
tsize = 2
#for vSize in vSizes:
#for tsize in tsizes:

model_name = 'fulldataset_patch' + str(vSize) + '_time' + str(tsize)

x_train,y_train,x_test,y_test = load_MaskCandidates(NumVideos=numvideos,voxel_Size=vSize,step=vSize,timeSize=tsize,timeStep=10,tol=0)

model=SqueezeNet(2, inputs=(x_train.shape[1:]))

#opt = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0001)
opt = keras.optimizers.SGD(lr=0.001,momentum = 0.9,decay=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)#,callbacks=callbacks_list)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
                                 featurewise_center=False, # set input mean to 0 over the dataset
                                 samplewise_center=False, # set each sample mean to 0
                                 featurewise_std_normalization=False, # divide inputs by std of the dataset
                                 samplewise_std_normalization=False, # divide each input by its std
                                 zca_whitening=False, # apply ZCA whitening
                                 rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
                                 width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
                                 horizontal_flip=True, # randomly flip images
                                 vertical_flip=False) # randomly flip images
    datagen.fit(x_train)
    model.fit_generator(datagen.flow(x_train, y_train,
                        batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test))

classes = model.predict(x_test, batch_size=batch_size)

y_pred = []
for i in range(classes.shape[0]):
    y_pred.append(np.argmax(classes[i, :]))
y_pred = np.array((y_pred))
target_names = ['Background', 'Cell']

labels_test = []
for i in range(y_test.shape[0]):
    labels_test.append(np.argmax(y_test[i, :]))
labels_test = np.array((labels_test))


classificationReport = classification_report(labels_test, y_pred, target_names=target_names)
print(classificationReport)

metrics_train = model.evaluate(x_train,y_train,batch_size=batch_size)
metrics_test = model.evaluate(x_test,y_test,batch_size=batch_size)

model.save(os.path.join(targetdir,model_name +'.h5'))
model.save_weights(os.path.join(targetdir,model_name +'_weights.h5'))
with open (os.path.join(targetdir,model_name + '.txt'),'w') as f:
  f.write('Parameters: \n Voxel_size: ' + str(vSize) + ' timeSize ' + str(tsize) + ' Step: ' + str(tsize) + '\n'
    + ' NumTrainSamples: '  + str(y_train.shape[0]) 
    + ' NumTestSamples: ' +str(y_test.shape[0])
    + ' NumVideos: ' + str(numvideos) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   +'\n' 
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))
