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
        circleShape = Point(gtCircles[aCircleGTIdx][0], gtCircles[aCircleGTIdx][1]).buffer(gtCircles[aCircleGTIdx][2])
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

    with open('/home/lapardo/SIPAIM/CellSegementation/celldivision/datos_stage/Stages_train.csv') as f:
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


baseDir = '/home/lapardo/SSD/alejo/MouEmbTrkDtb/'
numvideos = 100
frame_array, label_circles, Stages=getFramesandCircles(baseDir,numvideos,test=False)
def SemanticEval(params):
    bestGlobalParams = [params[0], 50, params[1], params[2], 140]
    totalJac = []
    for i in range(len(label_circles)):
        img = frame_array[i]
        gtCircles = label_circles[i]
        stage = Stages[i]
        circlesBase = getHoughCircles(img, bestGlobalParams)
        baseJaccard = calculateJaccard(circlesBase, gtCircles)
        totalJac.append(baseJaccard)
    totalJac = np.array(totalJac)
    meanJaccard = totalJac.mean()
    print('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')
    with open('../Hough/hough_CORRECT.csv','a') as f:
        f.write('mindists:' + str(params[0]) +' ,param2:' + str(params[1]) +' ,minRadius:' +  str(params[2]) + ' ,Mean Jaccard: ' + str(meanJaccard)+ '\n')

processPool = mp.Pool(20)
mindist = np.array((10,20,30,40,50,60))
param2 = np.array((10,15,20,25,30,35))
minRadius = np.array((30,35,40,45,50,55,60))
params = []
for d in mindist:
    for p in param2:
        for r in minRadius:
            params.append((d,p,r))

processPool.map(SemanticEval, params)