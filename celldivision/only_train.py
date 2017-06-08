from __future__ import print_function
import candidates as ca
import keras
from keras.layers import Activation
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling3D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
import random
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model = Sequential()

def step_decay(epoch):
  lrate = 0.01
  drop = 0.1
  if epochs == 75.0:
    lrate = lrate * drop
  return lrate

model.add(Conv3D(16, (1, 1, 1), padding='same',
          input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv3D(16, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Dropout(0.5))

model.add(Conv3D(16, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(16, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
#model.add(MaxPooling3D(pool_size=(2, 2)))#,padding = 'valid'))
model.add(Dropout(0.25))

model.add(Conv3D(32, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(32, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv3D(32, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(32, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Dropout(0.25))

model.add(Conv3D(64, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Conv3D(64, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(64, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Dropout(0.25))

model.add(Conv3D(128, (1, 1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv3D(128, (3, 3, 3),padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

opt =keras.optimizers.Adagrad(lr=0.00, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,callbacks=callbacks_list)
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
target_names = ['Background', 'Cell', 'Boundary']

classificationReport = classification_report(labels_test, y_pred, target_names=target_names)
print(classificationReport)

metrics_train = model.evaluate(x_train,y_train,batch_size=batch_size)
metrics_test = model.evaluate(x_test,y_test,batch_size=batch_size)

model.save(os.path.join(targetdir,model_name +'.h5'))
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
