import cv2
import numpy as np 
import math
import os
import time
import re
from shapely.geometry import Point
import multiprocessing as mp

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def getFramesandCircles(datapath,numvideos,test=False):

	numvideos = numvideos
	numtrain = int(numvideos*0.7)
	numtest = numvideos - numtrain

	videos = os.listdir(datapath)
	videos = videos[0:numvideos]

	trainvideos = videos[0:numtrain]
	trainvideos = sorted(trainvideos,key=natural_key)
	#print(trainvideos)
	testvideos = videos[numtrain:numtrain+numtest]
	testvideos = sorted(testvideos,key=natural_key)

	if test:
		videos = testvideos
	else:
		videos = trainvideos

	labels_array = np.zeros((0,21))
	frame_array = []
	for video in videos:

		pathL = os.path.join(datapath,video, '_trajectories.txt')
		with open(pathL) as f:
		        content = np.loadtxt(f)
		        #content = content[0:-1,:]
		labels_array = np.concatenate((labels_array,content[0:-1,:]))

		images = os.listdir(os.path.join(datapath,video))
		images = sorted(images,key=natural_key)
		images = images[0:len(content)-1]

		for image in images:
			if os.path.join(datapath,video,image).endswith('png'):
				print('Loading image ' + os.path.join(video,image))
				im = cv2.imread(os.path.join(datapath,video,image),0)
				frame_array.append(im)

	frame_array = np.array(frame_array)
	labels_array = np.array(labels_array)
	#print('frame_array',frame_array)
	label_image = []
	label_circles = []
	for i in range(labels_array.shape[0]):
		label_actual = labels_array[i]
		imagen_actual = frame_array[i,:,:]
		for pos in range(0,21):
			#print('pos',pos)
			if ((pos == 0) or ((pos)%3 == 0)) and (label_actual[pos] != 0):
				#print('pos ' + str(pos) + ' (pos+1)%3 ' + str((pos+1)%3) + ' label_actual[pos] '+str(label_actual[pos]))
				circle = label_actual[pos:pos+3]
				#print('circle',circle)
				label_image.append(circle)
		#Label circles tiene todo organizado por imagen
		label_circles.append(label_image)
		label_image = []
	return frame_array, label_circles

def JaccardInstance(im,gt,mindist,param2,minRadius,nms):
	#Predccion de hough
	nms = nms
	init_circles = cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,1,mindist,
	                        param1=50,param2=param2,minRadius=minRadius,maxRadius=140)

	#print('# of prediction for this one: ' + str(len(init_circles)))
	#print('init_circles',init_circles)
	#Aca arranca el NMS
	if init_circles is None:
		return 0
	else:
		pass
	init_circles = init_circles[0]
		
	circles = []
	circles.append(init_circles[0])
	for c in init_circles:
		a = Point(init_circles[0][1],init_circles[0][0]).buffer(init_circles[0][2])
		b = Point(c[1],c[0]).buffer(c[2])
		inter = a.intersection(b)
		union = a.union(b)
		Jaccard = inter.area/union.area
		if Jaccard < nms:
			circles.append(c)
	jac_perlabel = np.zeros((len(circles),len(gt)))


	if len(circles)>4:
		circles = circles[0:4]
	else:
		pass

	print(len(circles))

	j = 0 
	for pred in circles:
		k = 0
		for label in gt:
			real_circle = Point(label[1],label[0]).buffer(label[2])
			predicted_circle = Point(pred[1],pred[0]).buffer(pred[2])
			inter = real_circle.intersection(predicted_circle)
			union = real_circle.union(predicted_circle)
			Jaccard = inter.area/union.area
			#print('Jaccard',Jaccard)
			jac_perlabel[j,k] = Jaccard
			if Jaccard < 0.25:
				Jaccard = 0
			k+=1
		j += 1
	maximos_columna = []
	for index in range(jac_perlabel.shape[1]):
		maximos_columna.append(jac_perlabel[:,index].max())
		#jac_perlabel[:,index][jac_perlabel[:,index]!=jac_perlabel[:,index].max()]=0
	maximos_columna = maximos_columna.mean()

	maximos_fila = []
	for index in range(jac_perlabel.shape[0]):
		maximos_fila.append(jac_perlabel[index,:].max())
		#jac_perlabel[:,index][jac_perlabel[:,index]!=jac_perlabel[:,index].max()]=0
	maximos_fila = maximos_fila.mean()	

	jac_perpred = np.zeros((len(gt),len(circles)))

	k = 0 
	for label in gt:
		j = 0
		for pred in circles:
			real_circle = Point(label[1],label[0]).buffer(label[2])
			predicted_circle = Point(pred[1],pred[0]).buffer(pred[2])
			inter = real_circle.intersection(predicted_circle)
			union = real_circle.union(predicted_circle)
			Jaccard = inter.area/union.area
			#print('Jaccard',Jaccard)
			jac_perpred[k,j] = Jaccard
			j+=1
		k += 1
	for index in range(jac_perpred.shape[0]):
		jac_perpred[index,:][jac_perpred[index,:]!=jac_perpred[index,:].max()]=0
	return np.minimum(maximos_columna,maximos_fila)

#print('Applying Hough Transform to ' + str(ii) + ' of ' + str(frame_array.shape[0]))

datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 50
frame_array, label_circles = getFramesandCircles(datapath,numvideos,test=False)


def FinalJaccar(params):
	jac_perimage = []
	for ii in range(0,frame_array.shape[0]):
		print('Image ' + str(ii) + ' from ' + str(frame_array.shape[0]))
		im = cv2.medianBlur(frame_array[ii],5)
		gt = label_circles[ii]
		jac=JaccardInstance(im,gt,params[0],param2=params[1],minRadius=params[2],nms=params[3])
		jac_perimage.append(jac)
	jac_perimage = np.array(jac_perimage)
	meanJaccard = jac_perimage.mean()

	print('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,nms:'+ str(params[3]) +' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
	with open('../Hough/hough_CORRECT.csv','a') as f:
		f.write('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,nms:'+ str(params[3]) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')

processPool = mp.Pool(20)
mindist = np.array((10,20,30,40,50,60))
param2 = np.array((10,15,20,25,30,35))
minRadius = np.array((40,45,50,55,60))
nms = np.array((0.6,0.5,0.4,0.3,0.2))
params = []
for d in mindist:
	for p in param2:
		for r in minRadius:
			for nm in nms:
				params.append((d,p,r,nm))

processPool.map(FinalJaccar, params)
