from candidates import getStageLabel
import os
import cv2
import re


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

pathGT = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb/'
dataset = os.listdir(pathGT)
videos = sorted(dataset,key=natural_key)
frame_array_train = []
label_array_train = []
for video in dataset: 
	print('Video ' + video + ' from ' +str(len(dataset)))
	frames = os.listdir(os.path.join(pathGT	,video))
	frames = sorted(frames,key=natural_key)
	pathL = os.path.join(pathGT,video, '_trajectories.txt')
	with open(pathL) as f:
        content = np.loadtxt(f)
	frames = frames[0:len(content)]
	labels = getStageLabel(os.path.join(pathGT,video))
	for frame in frames:
		print('Frame ' + frame + ' from ' +str(len(frames)))
		if os.path.join(pathGT,video,frame).endswith('png'):
			im = cv2.imread(os.path.join(pathGT,video,frame))
		frame_array_train.append(im)
	label_array_train.append(labels)