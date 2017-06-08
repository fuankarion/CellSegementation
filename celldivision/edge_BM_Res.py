from __future__ import print_function
import candidates as ca
import keras
from keras.models import load_model
from keras.layers import Input, Activation, Conv3D, Dense, Dropout, Flatten, MaxPooling3D, BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models/3d/edge/'
model_name = 'BatchNorm_'
model_path = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models/3d/timesize/model_time2_fulldataset.h5'

batch_size = 1024
num_classes = 3
epochs = 100
freeze = 32
data_augmentation = False

datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 50
numtrain = int(numvideos*0.7)
numtest = numvideos - numtrain

videos = os.listdir(datapath)
videos = videos[0:numvideos]

trainvideos = videos[0:numtrain]
testvideos = videos[numtrain:numtrain+numtest]

voxelSize = 13
step = 15
timeSize = 2
tol = 5

voxel_array_train = []
labels_train = []
for i in range(0,numtrain):
  print('Video',trainvideos[i])
  videoCube_train = ca.loadVideoCube(os.path.join(datapath,trainvideos[i]))
  for x in range(0, videoCube_train.shape[0]-voxelSize, step):
      print('train_data', x)
      for y in range(0, videoCube_train.shape[1]-voxelSize, step):
          for z in range(0, videoCube_train.shape[3]-timeSize, step):
              #voxelDescriptor = getSTIPDescriptor(aVoxel)
              voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,trainvideos[i]))

              if voxelLabel == 0:
                     ignoreFlag = random.uniform(0.0, 1.0)
                     if ignoreFlag <= 1:
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
  for x in range(0, videoCube_test.shape[0]-voxelSize, step):
      print('test_data', x)
      for y in range(0, videoCube_test.shape[1]-voxelSize, step):
          for z in range(0, videoCube_test.shape[3]-timeSize, step):
              #voxelDescriptor = getSTIPDescriptor(aVoxel)
              voxelLabel = ca.getCubeLabel(x, y, z, tol, os.path.join(datapath,testvideos[i]))
              if voxelLabel == 0:
                  ignoreFlag = random.uniform(0.0, 1.0)
                  if ignoreFlag <= 1:
                      continue
              aVoxel = ca.getVoxelFromVideoCube(videoCube_test, x, y, z, voxelSize, timeSize)
              voxel_array_test.append(aVoxel)
              labels_test.append(voxelLabel)
              #labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)

x_test = np.array((voxel_array_test))
#x_test = np.expand_dims(x_test, axis=5)
y_test = np.array((labels_test))
y_test = keras.utils.to_categorical(labels_test, num_classes)

def step_decay(epoch):
  lrate = 0.001
  drop = 0.1
  if epochs == 75.0:
    lrate = lrate * drop
  return lrate

class_weight = {0 : 0,
    1: 1.,
    2: 5.}

x = Input(shape=x_train.shape[1:])

conv3d_1 = Conv3D(16,(1,1,1), padding='same', activation='relu')(x)
bm1_1 = BatchNormalization(axis=4)(conv3d_1)
conv3d_2 = Conv3D(16,(1,1,1), padding='same', activation='relu')(bm1_1)
bm1_2 = BatchNormalization(axis=4)(conv3d_2)
pool1 = MaxPooling3D(pool_size=(2, 2, 1))(bm1_2)

conv3d_3 = Conv3D(16,(1,1,1), padding='same', activation='relu')(pool1)
bm2_1 = BatchNormalization(axis=4)(conv3d_3)
conv3d_4 = Conv3D(16,(1,1,1), padding='same', activation='relu')(bm2_1)
bm2_2 = BatchNormalization(axis=4)(conv3d_4)

conv3d_5 = Conv3D(32,(1,1,1), padding='same', activation='relu')(bm2_2)
bm3_1 = BatchNormalization(axis=4)(conv3d_5)
conv3d_6 = Conv3D(32,(1,1,1), padding='same', activation='relu')(bm3_1)
bm3_2 = BatchNormalization(axis=4)(conv3d_6)

conv3d_7 = Conv3D(32,(1,1,1), padding='same', activation='relu')(bm3_2)
bm4_1 = BatchNormalization(axis=4)(conv3d_7)
conv3d_8 = Conv3D(32,(1,1,1), padding='same', activation='relu')(bm4_1)
bm4_2 = BatchNormalization(axis=4)(conv3d_8)
pool2 = MaxPooling3D(pool_size=(2, 2, 1))(bm4_2)

conv3d_9 = Conv3D(64,(1,1,1), padding='same', activation='relu')(pool2)
bm5_1 = BatchNormalization(axis=4)(conv3d_9)
conv3d_10 = Conv3D(64,(1,1,1), padding='same', activation='relu')(bm5_1)
bm5_2 = BatchNormalization(axis=4)(conv3d_10)

conv3d_11 = Conv3D(64,(1,1,1), padding='same', activation='relu')(bm5_2)
bm6_1 = BatchNormalization(axis=4)(conv3d_11)
conv3d_12 = Conv3D(64,(1,1,1), padding='same', activation='relu')(bm6_1)
bm6_2 = BatchNormalization(axis=4)(conv3d_12)
pool3 = MaxPooling3D(pool_size=(2, 2, 1))(bm6_2)

conv3d_13 = Conv3D(64,(1,1,1), padding='same', activation='relu')(pool3)
bm7_1 = BatchNormalization(axis=4)(conv3d_13)
conv3d_14 = Conv3D(64,(1,1,1), padding='same', activation='relu')(bm7_1)
bm7_2 = BatchNormalization(axis=4)(conv3d_14)

flatten_1 = Flatten()(bm7_2)
dense_1 = Dense(256,activation='relu')(flatten_1)
drop1 = Dropout(0.5)(dense_1)
dense_2 = Dense(num_classes,activation='softmax')(drop1)

model = Model(inputs=x,outputs=dense_2)

#model.load_weights('/home/lapardo/SIPAIM/CellSegementation/celldivision/models/3d/timesize/weights_time2_fulldataset.h5',by_name=True)

opt = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

if not data_augmentation:

  print('Start Training')
  start = time.time()
  model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_test, y_test),
            shuffle=True,class_weight = class_weight)
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

"""
class_weight = {0 : 0,
    1: 50.,
    2: 1.}
"""
classes = model.predict(x_test, batch_size=64)
y_pred = []

for i in range(classes.shape[0]):
    y_pred.append(np.argmax(classes[i, :]))
y_pred = np.array((y_pred))
target_names = ['Cell', 'Boundary', 'Background']
classificationReport = classification_report(labels_test, y_pred, target_names=target_names)
print(classificationReport)

metrics_train = model.evaluate(x_train,y_train,batch_size=batch_size)
metrics_test = model.evaluate(x_test,y_test,batch_size=batch_size)

model.save(os.path.join(targetdir,model_name +'.h5'))
with open (os.path.join(targetdir,model_name + '.txt'),'w') as f:
  f.write('Parameters: \n Voxel_size: ' + str(voxelSize) + ' timeSize ' + str(timeSize) + ' Step: ' + str(step) + ' tol: ' + str(tol) + ' Freeze: ' + str(freeze) + '\n'
    + ' NumTrainSamples: '  + str(len(voxel_array_train)) 
    + ' NumTestSamples: ' +str(len(voxel_array_test))
    + ' NumVideos: ' + str(numvideos) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   +'\n' 
    + ' Batch Size: ' +  str(batch_size) + '\n'
    + ' Train Time: ' + str(elapsed_time) + '\n' 
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))

