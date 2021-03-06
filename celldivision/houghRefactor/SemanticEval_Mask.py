import cv2
import numpy as np
import os
from shapely.geometry import Point
from shapely.ops import cascaded_union
import re
import multiprocessing as mp

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def getHoughCircles(img, paramss):
    init_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=paramss[0], 
                               param1=paramss[1], param2=paramss[2], 
                               minRadius=paramss[3], maxRadius=paramss[4])
    if init_circles is None:
        return init_circles
    else:
        pass

    circles = []
    circles.append(init_circles[0][0])
    for c in init_circles[0]:
        a = Point(init_circles[0][0][1],init_circles[0][0][0]).buffer(init_circles[0][0][2])
        b = Point(c[1],c[0]).buffer(c[2])
        inter = a.intersection(b)
        union = a.union(b)
        Jaccard = inter.area/union.area
        if Jaccard < 0.5:
            circles.append(c)

    if len(circles)>4:
        circles = circles[0:4]
    else:
        pass

    return circles

def drawCircles(img, circles):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def loadGT(video):
    contentGT = None
    pathGT = os.path.join(baseDir, video, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
    return contentGT

def loadAndPrepocessImage(imPath):
    img = cv2.imread(imPath, 0)
    img = cv2.medianBlur(img, 5)
    return img

def getCircleLabelsForFrame(video, frameIdx):
    contentGT = loadGT(video)
    gtLine = contentGT[frameIdx]
    tokens = gtLine.split('\t')
    
    gtCircles = []
    for tokensIdx in range(0, len(tokens)-3, 3):
        cellRadius = int(tokens[tokensIdx + 2])
        if  cellRadius > 0:#Cell candidate
            circle = [int(tokens[tokensIdx + 1]), int(tokens[tokensIdx]), cellRadius]
            gtCircles.append(circle)
            
    return gtCircles
    
def calculateJaccard(circles, gtCircles):
    #Better jaccard Calculation

    gtAsShape = []
    for aCircleGTIdx in range(len(gtCircles)):
        circleShape = Point(gtCircles[aCircleGTIdx][1], gtCircles[aCircleGTIdx][0]).buffer(gtCircles[aCircleGTIdx][2])
        gtAsShape.append(circleShape)
    gtUnion = cascaded_union(gtAsShape)

    predsAsShape = []
    if circles is None:
        circleShape = Point(0, 0).buffer(1)
        predsAsShape.append(circleShape)
    else:
        for aCirclePredIdx in range(len(circles)):
            #if aCirclePredIdx>1:
            #    break;
            circleShape = Point(circles[aCirclePredIdx][0], circles[aCirclePredIdx][1]).buffer(circles[aCirclePredIdx][2])
            predsAsShape.append(circleShape)

    predsUnion = cascaded_union(predsAsShape)

    inter = predsUnion.intersection(gtUnion)
    union = predsUnion.union(gtUnion)
    jaccard = inter.area / union.area
    return jaccard

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
    masks = []
    for video in videos:

        pathL = os.path.join(datapath,video, '_trajectories.txt')
        with open(pathL) as f:
                content = np.loadtxt(f)
                #content = content[0:-1,:]
        labels_array = np.concatenate((labels_array,content[0:-1,:]))

        images = os.listdir(os.path.join(datapath,video))
        images = sorted(images,key=natural_key)
        images = images[0:len(content)-1]
	idx = 0
        for image in images:
            idx += 1
            if os.path.join(datapath,video,image).endswith('png'):
                print('Loading image ' + os.path.join(video,image))
                im = cv2.imread(os.path.join(datapath,video,image),0)
                im = cv2.medianBlur(im,5)
		mask = cv2.imread(os.path.join('/home/lapardo/SSD/FilteredMasks',video,str(idx)+'.jpg'),0)
		print('mask',os.path.join('/home/lapardo/SSD/FilteredMasks',video,str(idx)+'.jpg'))
		mask = mask/255
		im = np.multiply(im,mask)
		Stages.append(stages_dict[os.path.join(video,image)])
                frame_array.append(im)
		masks.append(mask)
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


baseDir = '/home/lapardo/SSD/MouEmbTrkDtb/'
numvideos = 100
frame_array_complete, label_circles_complete, Stages=getFramesandCircles(baseDir,numvideos,test=True)
#target_stage = 1
frame_array = []
label_circles = []
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
    frame_array.append(frame_array_complete[n])
    label_circles.append(label_circles_complete[n])
frame_array = np.array(frame_array)

def SemanticEval():
    bestGlobalParams = [params[0], 50, params[1], params[2], 140]
    totalJac = []
    for i in range(len(label_circles)):
        img = frame_array[i]
        gtCircles = label_circles[i]
        stage = Stages[i]
        if stage == 0:
                bestGlobalParams = [20, 50, 25, 60, 140]
        elif stage == 1:
                bestGlobalParams = [40, 50, 30, 100, 140]
        elif stage == 2:
                bestGlobalParams = [20, 50, 30, 70, 140]
        elif stage == 1:
                bestGlobalParams = [20, 50, 25, 60, 140]
        else:
                bestGlobalParams = [20, 50, 25, 60, 140]
        circlesBase = getHoughCircles(img, bestGlobalParams)
        baseJaccard = calculateJaccard(circlesBase, gtCircles)
        totalJac.append(baseJaccard)
    totalJac = np.array(totalJac)
    meanJaccard = totalJac.mean()
    print('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
    with open('../Hough/Stage_hough_CORRECT.txt','a') as f:
        f.write(str(meanJaccard)+ '\n')

SemanticEval()
