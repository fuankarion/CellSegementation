from __future__ import print_function
import candidates as ca
import keras
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from Squeeze_2D_Mask import SqueezeNet, load_MaskCandidates
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import os
from random import randint
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

weights_dir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/textures/second_model_decay1e-4.h5'

targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models_correct/ftTextures/'
model_name = 'FineTuningTextures'

batch_size = 1024
num_classes = 2
epochs = 100
freeze = 0

x_train,y_train,x_test,x_train = load_MaskCandidates(NumVideos=50,voxel_Size=13,step=13,timeSize=1,timeStep=2,tol=0)

model = SqueezeNet(2)

model.load_weights(weights_dir)

for layer in model.layers[0:freeze]:
        layer.trainable = False

opt = keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.0001)
#opt = keras.optimizers.SGD(lr=0.001,momentum = 0.9,decay=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]


print('Not using data augmentation.')
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
  f.write('Parameters: \n Voxel_size: ' + str(voxelSize) + ' timeSize ' + str(timeSize) + ' Step: ' + str(step) + ' tol: ' + str(tol) + '\n'
    + ' NumTrainSamples: '  + str(len(voxel_array_train)) 
    + ' NumTestSamples: ' +str(len(voxel_array_test))
    + ' NumVideos: ' + str(numvideos) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   +'\n' 
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))

