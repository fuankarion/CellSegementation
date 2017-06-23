from __future__ import print_function
import candidates as ca
import keras
from keras.models import load_model, Model
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, add, MaxPooling2D, Input, AveragePooling2D, Concatenate, AveragePooling2D
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import os
from random import randint
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

weights_dir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/textures/third_model_lrstep_weights.h5'

targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models_stage/ftTextures/'
model_name = 'FineTuningTextures'

batch_size = 1024
num_classes = 4
epochs = 100
freeze = 21

da = np.load('datos_stage/StageData_128.npz')

x_train = da['x_train']
y_train = da['y_train']
x_test = da['x_test']
y_test = da['y_test']

def SqueezeNet(nb_classes, inputs=(128,128,3)):
    """ Keras Implementation of SqueezeNet(arXiv 1602.07360)
    @param nb_classes: total number of final categories
    Arguments:
    inputs -- shape of the input images (channel, cols, rows)
    """
    input_img = Input(shape=inputs)#1
    conv1 = Conv2D(#2
        32, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        strides=(2,2), padding='same', name='conv1')(input_img)
    maxpool1 = MaxPooling2D(#3
        pool_size=(3,3), strides=(2,2), name='maxpool1')(conv1)

    fire2_squeeze = Conv2D(#4
        8, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Conv2D(#5
        16, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Conv2D(#6
        16, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = Concatenate(axis=-1)([fire2_expand1, fire2_expand2])#7

    fire3_squeeze = Conv2D(#8
        8, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Conv2D(#9
        16, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Conv2D(#10
        16, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = Concatenate(axis=-1)([fire3_expand1, fire3_expand2])#11

    residual32 = add([merge2, merge3])

    fire4_squeeze = Conv2D(#12
        16, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_squeeze')(residual32)
    fire4_expand1 = Conv2D(#13
        32, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Conv2D(#14
        32, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = Concatenate(axis=-1)([fire4_expand1, fire4_expand2])#15
    maxpool4 = MaxPooling2D(#16
        pool_size=(3,3), strides=(2,2), name='maxpool4')(merge4)

    fire5_squeeze = Conv2D(#17
        16, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Conv2D(#18
        32, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Conv2D(#19
        32, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = Concatenate(axis=-1)([fire5_expand1, fire5_expand2])#20

    residual45 = add([maxpool4, merge5])

    fire6_squeeze = Conv2D(#21
        24, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Conv2D(#22
        64, (1,1), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Conv2D(#23
        64, (3,3), activation='relu', kernel_initializer='glorot_uniform',
        padding='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = Concatenate(axis=-1)([fire6_expand1, fire6_expand2])#24

    fire6_dropout = Dropout(0.5, name='fire6_dropout')(merge6)#25
    conv10 = Conv2D(#26
        nb_classes, (1,1), kernel_initializer='glorot_uniform',
        padding='valid', name='conv10_4')(fire6_dropout)

    # The size should match the output of conv10
    avgpool10 = AveragePooling2D(#27
        (15, 15), name='avgpool10_4')(conv10)

    flatten = Flatten(name='flatten_4')(avgpool10)#28
    softmax = Activation("softmax", name='softmax_4')(flatten)#29

    return Model(inputs=input_img, outputs=softmax)


model = SqueezeNet(num_classes,inputs=(x_train.shape[1:]))

model.load_weights(weights_dir, by_name=True)

for layer in model.layers[0:freeze]:
        layer.trainable = False

def step_decay(epoch):
  lrate_initial = 0.001
  drop = 0.1
  if epoch >= 35:
    lrate = lrate_initial * drop
  else:
    lrate = lrate_initial
  print('lrate',lrate)
  return lrate

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

#opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0001)
#opt = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0001)
opt = keras.optimizers.SGD(lr=0.001,momentum = 0.9,decay=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

print('Start Training')
start = time.time()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)#,callbacks=callbacks_list)
end = time.time()
elapsed_time = end-start
print('Train time', elapsed_time)

classes = model.predict(x_test, batch_size=batch_size)

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
  f.write(' NumVideos: ' + str(100) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   + '\n'
    + ' Batch_size: ' + str(batch_size) + '\n' 
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))
