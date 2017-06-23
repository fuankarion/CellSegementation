import os 
import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp
import scipy.misc

def MedianFilter(args):
	source = args[0]
	video = args[1]
	im = args[2]
	print('Processing: ' + source + im)
	targetdir = '/home/lapardo/SSD/alejo/FilteredMasks/'
	oldmask = cv2.imread(os.path.join(source,video,im))
	newmask = cv2.medianBlur(oldmask,31)
	if not os.path.exists(os.path.join(targetdir,video)):
		os.mkdir(os.path.join(targetdir,video))
		scipy.misc.imsave(os.path.join(targetdir,video,im),newmask)
	else:
		scipy.misc.imsave(os.path.join(targetdir,video,im),newmask)


source = '/home/lapardo/SSD/alejo/Segmentations/'
videos = os.listdir(source)

args = []
for video in videos:
	frames = os.listdir(os.path.join(source,video))
	for mask in frames:
		args.append((source,video,mask))

processPool = mp.Pool(20)
processPool.map(MedianFilter,args)
print('Done')