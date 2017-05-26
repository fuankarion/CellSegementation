from __future__ import print_function
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout, GlobalAveragePooling2D, \
    warnings
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.utils import get_file
from keras.utils import layer_utils
from get_candidates import get_candidates
from squeeze_net2D import SqueezeNet
import candidates as ca
import os
import random
import numpy as np
import keras

targetdir = '/home/lapardo/SIPAIM/CellSegementation/celldivision/models/Squeeze'
model_name = 'Squeeze'

batch_size = 8192
num_classes = 3
epochs = 100


voxelSize = 13
step = 15
timeSize = 1
tol = 0
numvideos = 30 

x_train,y_train,x_test,y_test,labels_test,labels_train = get_candidates(voxelSize,step,timeSize,tol,numvideos = numvideos)

model = SqueezeNet(input_tensor = x_train, input_shape=x_train.shape)

opt =keras.optimizers.Adagrad(lr=0.001, epsilon=1e-08, decay=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True)


classes = model.predict(x_test, batch_size=64)

y_pred = []
for i in range(classes.shape[0]):
    y_pred.append(np.argmax(classes[i, :]))

y_pred = np.array((y_pred))
target_names = ['Background', 'Cell', 'Boundary']

classificationReport = classification_report(labels_test, y_pred, target_names=target_names)
print(classificationReport)

metrics_train = model.evaluate(x_train,y_train,batch_size=batch_size)
metrics_test = model.evaluate(x_test,y_test,batch_size=batch_size)

with open (os.path.join(targetdir,model_name + '.txt'),'w') as f:
  f.write('Parameters: \n Voxel_size: ' + str(voxelSize) + ' Step: ' + str(step) + ' tol: ' + str(tol) + '\n'
    + ' NumTrainSamples: '  + str(len(voxel_array_train)) 
    + ' NumTestSamples: ' +str(len(voxel_array_test))
    + ' NumVideos: ' + str(numvideos) +'\n'
    + ' loss_train: ' + str(metrics_train[0]) + ' Acc_train: ' + str(metrics_train[1]) +'\n'
    + ' loss_test: ' + str(metrics_test[0]) + ' Acc_test: ' + str(metrics_test[1]) +'\n'
    + ' Epochs: ' + str(epochs)   +'\n' 
    + ' Classification_Report: \n' + classificationReport + '\n'
    + ' Model_dir: ' + os.path.join(targetdir,model_name + '.h5'))

model.save(os.path.join(targetdir,model_name +'.h5'))