import cv2
import numpy as np 
import math
import os
import time
import re
#from matplotlib import pyplot as plt
from shapely.geometry import Point
import multiprocessing as mp

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
maskpath = '/home/lapardo/SSD/alejo/FilteredMasks/'
numvideos = 100
numtrain = int(numvideos*0.7)
numtest = numvideos - numtrain

videos = os.listdir(datapath)
videos = videos[0:numvideos]

trainvideos = videos[0:numtrain]
testvideos = videos[numtrain:numtrain+numtest]
testvideos = sorted(testvideos,key=natural_key)

#Load frames and masks

con = 0
frame_array = []
mask_array = []
labels_array = np.zeros((0,21))

for video in testvideos:
	#print('Test: Video ' + str(con+1) + ' from ' +str(len(testvideos)))
	frames = os.listdir(os.path.join(datapath ,video))
	frames = sorted(frames,key=natural_key)

	masks = os.listdir(os.path.join(maskpath ,video))
	masks = sorted(masks,key=natural_key)	

	pathL = os.path.join(datapath,video, '_trajectories.txt')
	with open(pathL) as f:
	        content = np.loadtxt(f)
	        #content = content[0:-1,:]
	labels_array = np.concatenate((labels_array,content[0:-1,:]))
	frames = frames[0:len(content)-1]
	mask_number = 0
	for frame in frames:
		mask_number += 1
		print('Test: Frame ' + frame + ' from ' +str(len(frames)))
		if os.path.join(datapath,video,frame).endswith('png'):
			im = cv2.imread(os.path.join(datapath,video,frame),0)
			mask = cv2.imread(os.path.join(maskpath,video,str(mask_number)+'.jpg'),0)
			#im = cv2.resize(im,(180,180))
		frame_array.append(im)
		mask_array.append(mask)
	con += 1
frame_array = np.array(frame_array)
mask_array = np.array(mask_array)
#frame_array = frame_array[0:-1]
labels_array = np.array(labels_array)

#Generate cells annotations
label_image = []
label_circles = []

for i in range(0,labels_array.shape[0]):
	label_actual = labels_array[i]
	imagen_actual = frame_array[i,:,:]
	for pos in range(0,21):
		#print('pos',pos)
		if ((pos == 0) or ((pos)%3 == 0)) and (label_actual[pos] != 0):
			#print('pos ' + str(pos) + ' (pos+1)%3 ' + str((pos+1)%3) + ' label_actual[pos] '+str(label_actual[pos]))
			circle = label_actual[pos:pos+3]
			print('circle',circle)
			label_image.append(circle)
	label_circles.append(label_image)
	label_image = []


#Load Predicted Stages
arch = '/home/lapardo/SIPAIM/CellSegementation/celldivision/datos_stage/Stages.csv'
with open(arch) as f:
	contents = f.readlines()
contents = [x.replace('\n','') for x in contents]
contents = [x.split(',') for x in contents]
stages = np.array([int(x[2]) for x in contents])
stages[stages==0]=3 #transition stage
stages[stages==3]=4 #third stage with four cells

radius = []
#Generate radius biggest circle
for mask in mask_array:
	mask[mask>1] = 1
	nonZero = np.count_nonzero(mask)
	ra = math.sqrt(nonZero / math.pi)
	#print('radius',ra)
	radius.append(ra)
radius = np.array(radius)

def Jaccard_Calculation(parametros):
	mindist = parametros[0]
	param2 = parametros[1]
	minRadius = parametros[2]
	nms = parametros[3]
	scores_TPFP = []
	scores_FN = []
	start = time.time()
	for ii in range(0,frame_array.shape[0]):
		print('Image ' + str(ii) + ' from ' + str(frame_array.shape[0]))
		stage = stages[ii]
		im = cv2.medianBlur(frame_array[ii],5)
		#print('Applying Hough Transform to ' + str(ii) + ' of ' + str(frame_array.shape[0]))
		start = time.time()
		init_circles = cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,1,mindist,
	                            param1=50,param2=param2,minRadius=minRadius,maxRadius=140)
		end = time.time()
		#print('Hough transform done, time elapsed: ' + str(end-start))
		label_actual = label_circles[ii]
		if init_circles is None:
			jac_perpred = []		
			for i in range(0,len(label_actual)):
				jac_perpred.append(0)
			jac_perpred = np.array(jac_perpred)
			#print('jacc',jac_perpred)
			scores_TPFP.append(jac_perpred.mean())
			continue
		init_circles = init_circles[0]
		#print('init_circles',init_circles)
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
		#label_actual = label_circles[ii]
		jac_perlabel = np.zeros((len(circles),len(label_actual)))
		j = 0 
		for pred in circles:
			k = 0
			#jac_perpred = np.zeros(len(circles[0]))
			for label in label_actual:
				real_circle = Point(label_actual[0][1],label_actual[0][0]).buffer(label_actual[0][2])
				predicted_circle = Point(pred[1],pred[0]).buffer(pred[2])
				inter = real_circle.intersection(predicted_circle)
				union = real_circle.union(predicted_circle)
				Jaccard = inter.area/union.area
				#print('Jaccard',Jaccard)
				jac_perlabel[j,k] = Jaccard
				k+=1
			j += 1
		jac_perpred = []
		for f in range(0,jac_perlabel.shape[0]):
			jac_perpred.append(jac_perlabel[f].max()) 
		jac_perpred = np.array(jac_perpred)
		scores_TPFP.append(jac_perpred.mean())
		jac_perpred = []
	for ii in range(0,frame_array.shape[0]):
		print('Label vs Pred: Image ' + str(ii) + ' 	from ' + str(frame_array.shape[0]))
		im = cv2.medianBlur(frame_array[ii],5)
		init_circles = cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,1,mindist,
	                            param1=50,param2=param2,minRadius=minRadius,maxRadius=140)
		label_actual = label_circles[ii]
		if init_circles is None:
			jac_perpred = []		
			for i in range(0,len(label_actual)):
				jac_perpred.append(0)
			jac_perpred = np.array(jac_perpred)
			#print('jacc',jac_perpred)
			scores_FN.append(jac_perpred.mean())
			continue
		init_circles = init_circles[0]	
		circles = []
		circles.append(init_circles[0])
		for c in init_circles:
			#print('initcircle',init_circles[0])
			#print('c',c)
			a = Point(init_circles[0][1],init_circles[0][0]).buffer(init_circles[0][2])
			b = Point(c[1],c[0]).buffer(c[2])
			inter = a.intersection(b)
			union = a.union(b)
			Jaccard = inter.area/union.area
			#print('Jaccard',Jaccard)
			if Jaccard < nms:
				circles.append(c)
		#label_actual = label_circles[ii]
		jac_perlabel = np.zeros((len(label_actual),len(circles)))
		j = 0 
		for label in label_actual:
			k = 0
			#jac_perpred = np.zeros(len(circles[0]))
			for pred in circles:
				real_circle = Point(label_actual[0][1],label_actual[0][0]).buffer(label_actual[0][2])
				predicted_circle = Point(pred[1],pred[0]).buffer(pred[2])
				inter = real_circle.intersection(predicted_circle)
				union = real_circle.union(predicted_circle)
				Jaccard = inter.area/union.area
				#print('Jaccard',Jaccard)
				jac_perlabel[j,k] = Jaccard
				k+=1
			j += 1
		jac_perpred = []
		for f in range(0,jac_perlabel.shape[0]):
			jac_perpred.append(jac_perlabel[f].max()) 
		jac_perpred = np.array(jac_perpred)
		#print('jacc',jac_perpred)
		scores_FN.append(jac_perpred.mean())
		jac_perpred = []
	end = time.time()
	print('Elapsed Time',end - start)
	print('Computing Final Jaccard')
	scores_FN = np.array(scores_FN)
	scores_TPFP = np.array(scores_TPFP)
	Final_Jaccards = np.minimum(scores_FN,scores_TPFP)
	meanJaccard = Final_Jaccards.mean()
	print('Final Jaccard: ' + str(meanJaccard))
	#print('meanJaccard',meanJaccard)
	print('mindists:' + str(mindist) +' ,param2:' + str(param2) +' ,minRadius:' +  str(minRadius) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
	with open('Hough/Complete/hough_paramsExploration_correct-NMS.csv','a') as f:
		f.write('mindists:' + str(mindist) +' ,param2:' + str(param2) +' ,minRadius:' +  str(minRadius) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
	scores_FN = []
	scores_TPFP = []

processPool = mp.Pool(20)

mindists = np.array((10,20,30,40,50,60))
param2 = np.array((15,20,25,30,35,40,45))
minRadius = np.array((40,45,50,55,60,65,70))

parametros = []

for dist in mindists:
	for param in param2:
		for radius in minRadius:
			parametros.append((dist,param,radius))

#Jaccard_Calculation(parametros[0])
processPool.map(Jaccard_Calculation, parametros)
#Jaccard_Calculation((10,30,70,0.6))