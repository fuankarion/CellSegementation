from candidates import getStageLabel
import re
import cv2
import time
import keras
import numpy as np
import os

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

batch_size = 1024
num_classes = 4
epochs = 100
data_augmentation = False

image_size = 180

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
names_train = []
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
			im = cv2.resize(im,(image_size,image_size))
		frame_array_train.append(im)
		names_train.append(video+'/'+frame)
	label_array_train.append(labels)

labels_train = []
for video_labels in label_array_train:
	labels_train = np.concatenate((labels_train,video_labels))

x_train = np.array(frame_array_train)
y_train = y_train = keras.utils.to_categorical(labels_train, num_classes)

frame_array_test = []
label_array_test = []
names_test = []
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
			im = cv2.resize(im,(image_size,image_size))
		frame_array_test.append(im)
		names_test.append(video+'/'+frame)
	label_array_test.append(labels)

labels_test = []
for video_labels in label_array_test:
	labels_test = np.concatenate((labels_test,video_labels))

x_test = np.array(frame_array_test)
y_test = keras.utils.to_categorical(labels_test, num_classes)

np.savez('/home/lapardo/SIPAIM/CellSegementation/celldivision/datos_stage/StageData_180_withnames',x_train=x_train,y_train=y_train,names_train=names_train,x_test=x_test,y_test=y_test,names_test=names_test)