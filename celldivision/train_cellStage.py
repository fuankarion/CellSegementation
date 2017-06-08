from candidates import getStageLabel
import os
import cv2

pathGT = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb/'
dataset = os.listdir('/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb/')
frame_array_train = []
label_array_train = []

for video in dataset: 
	print('Video ' + video + ' from ' +str(len(dataset)))
	frames = os.listdir(os.path.join(pathGT	,video))
	for frame in frames:
		print('Frame ' + frame + ' from ' +str(len(frames)))
		if os.path.join(pathGT,video,frame).endswith('png'):
			im = cv2.imread(os.path.join(pathGT,video,frame))
			labels = getStageLabel(os.path.join(pathGT,video))
	frame_array_train.append(im)
	label_array_train.append(labels)