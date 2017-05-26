from __future__ import print_function
import keras
from keras.applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, Convolution3D, MaxPooling2D, MaxPooling3D, Activation, concatenate, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report
import tensorflow as tf
import random


def fire_module3D(x, fire_id, squeeze=16, expand=64):
	
	sq1x1 = "squeeze1x1"
	exp1x1 = "expand1x1"
	exp3x3 = "expand3x3"
	relu = "relu_"

    s_id = 'fire' + str(fire_id) + '/'

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    
    x = Convolution3D(squeeze, (1, 1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Convolution3D(expand, (1, 1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Convolution3D(expand, (3, 3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')
    return x

