from __future__ import print_function
import candidates as ca
import os
import random
import numpy as np
import keras

def get_candidates(voxelSize,step,timeSize,tol,numvideos = 30):

	datapath = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb/'
	numvideos = numvideos
	numtrain = int(numvideos*0.7)
	numtest = numvideos - numtrain
	num_classes = 3

	videos = os.listdir(datapath)
	videos = videos[0:numvideos]

	trainvideos = videos[0:numtrain]
	testvideos = videos[numtrain:numtrain+numtest]

	voxelSize = voxelSize
	step = step
	timeSize = timeSize
	tol = tol

	voxel_array_train = []
	labels_train = []

	for i in range(0,numtrain):
	  print('Video',trainvideos[i])
	  videoCube_train = ca.loadVideoCube(os.path.join(datapath,trainvideos[i]))
	  for x in range(0, videoCube_train.shape[0]-voxelSize, step):
	      print('train_data', x)
	      for y in range(0, videoCube_train.shape[1]-voxelSize, step):
	          for z in range(0, videoCube_train.shape[2]-timeSize, step):
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
	          for z in range(0, videoCube_test.shape[2]-timeSize, step):         
	              #voxelDescriptor = getSTIPDescriptor(aVoxel)
	              voxelLabel = ca.getCubeLabel(x, y, z, tol,os.path.join(datapath,testvideos[i]))
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

	return x_train,y_train,x_test,y_test,labels_test,labels_train