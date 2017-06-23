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
data_augmentation = False

x_train,y_train,x_test,x_train = load_MaskCandidates(NumVideos=50,voxel_Size=13,step=13,timeSize=1,timeStep=2,tol=0)

model = SqueezeNet(2)

model.load_weights(weights_dir)