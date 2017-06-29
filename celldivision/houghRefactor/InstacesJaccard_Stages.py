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

    trainvideos = ['E00', 'E01', 'E02', 'E03', 'E04', 'E06', 'E08', 'E09', 'E10',
       'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E18', 'E19', 'E21',
       'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E32', 'E33', 'E37',
       'E38', 'E40', 'E41', 'E42', 'E45', 'E46', 'E47', 'E48', 'E50',
       'E51', 'E55', 'E56', 'E57', 'E61', 'E62', 'E63', 'E65', 'E66',
       'E68', 'E69', 'E70', 'E73', 'E74', 'E75', 'E76', 'E77', 'E78',
       'E80', 'E81', 'E82', 'E83', 'E84', 'E85', 'E86', 'E88', 'E89',
       'E90', 'E91', 'E92', 'E94', 'E95', 'E98', 'E99']
    trainvideos = trainvideos[0:numtrain]

    #print(trainvideos)
    testvideos = ['E05', 'E07', 'E17', 'E20', 'E22', 'E23', 'E24', 'E31', 'E34',
       'E35', 'E36', 'E39', 'E43', 'E44', 'E49', 'E52', 'E53', 'E54',
       'E58', 'E59', 'E60', 'E64', 'E67', 'E71', 'E72', 'E79', 'E87',
       'E93', 'E96', 'E97']
    testvideos = testvideos[0:numtest]


    if test:
        videos = testvideos
	stagesfile = 'Stages_test'
    else:
        videos = trainvideos
	stagesfile = 'Stages_train'

	with open('/home/lapardo/SIPAIM/CellSegementation/celldivision/datos_stage/'+stagesfile+'.csv') as f:
		lines = f.readlines()
	lines = [x.replace('\n','') for x in lines]
	lines = [x.split(',') for x in lines]
	names = [x[0] for x in lines]
	stage_numbers = [int(x[1]) for x in lines]
	stages_dict = dict(zip(names,stage_numbers))

	labels_array = np.zeros((0,21))
	frame_array = []
	Stages = []
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
				Stages.append(stages_dict[os.path.join(video,image)])
				frame_array.append(im)

	frame_array = np.array(frame_array)
	Stages = np.array(Stages)
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

	return frame_array, label_circles, Stages

def JaccardInstance(im,gt,mindist,param2,minRadius,nms):
	#Predccion de hough
	nms = nms
	init_circles = cv2.HoughCircles(im,cv2.HOUGH_GRADIENT,1,mindist,
	                        param1=50,param2=param2,minRadius=minRadius,maxRadius=140)
	if init_circles is None:
		return 0
	else:
		pass

	if len(init_circles[0])>4:
		init_circles = init_circles[0][0:4]
	else:
		init_circles = init_circles[0]

	print(len(init_circles))
	#print('# of prediction for this one: ' + str(len(init_circles)))
	#print('init_circles',init_circles)
	#Aca arranca el NMS
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
			k+=1
		j += 1
	for index in range(jac_perlabel.shape[1]):
		jac_perlabel[:,index][jac_perlabel[:,index]!=jac_perlabel[:,index].max()]=0

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
	return jac_perpred.mean()

#print('Applying Hough Transform to ' + str(ii) + ' of ' + str(frame_array.shape[0]))

datapath = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 100
frame_array, label_circles, Stages = getFramesandCircles(datapath,numvideos,test=False)
for n in range(len(label_circles_complete)):
    if Stages[n] == 0:
	params = ((20,35,70))
    elif Stages[n] == 1:
	params = ((40,30,100))
    elif Stages[n] == 2:
	params = ((20,30,70))
    elif Stages[n] == 3:
	params = ((40,25,60))
    else:
	params = ((20,65,70))

def FinalJaccar():
	jac_perimage = []
	for ii in range(0,frame_array.shape[0]):
		print('Image ' + str(ii) + ' from ' + str(frame_array.shape[0]))
		im = cv2.medianBlur(frame_array[ii],5)
		gt = label_circles[ii]
		jac=JaccardInstance(im,gt,params[0],param2=params[1],minRadius=params[2],nms=0.5)
		jac_perimage.append(jac)
	jac_perimage = np.array(jac_perimage)
	meanJaccard = jac_perimage.mean()

	print('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,nms:'+ str(params[3]) +' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
	with open('../Hough/hough_CORRECT.csv','a') as f:
		f.write('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,nms:'+ str(params[3]) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')

#processPool = mp.Pool(20)
#mindist = np.array((10,20,30,40,50,60))
#param2 = np.array((10,15,20,25,30,35))
#minRadius = np.array((40,45,50,55,60))
#nms = np.array((0.6,0.5,0.4,0.3,0.2))
#params = []
#for d in mindist:
#	for p in param2:
#		for r in minRadius:
#			for nm in nms:
#				params.append((d,p,r,nm))

#processPool.map(FinalJaccar, params)
