import cv2
import numpy as np
import os
from shapely.geometry import Point
from shapely.ops import cascaded_union

baseDir = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb/'

bestGlobalParams = [35, 50, 35, 70, 140]

params1Cell = [35, 50, 30, 90, 140]
params2Cell = [35, 50, 20, 60, 110]
params3Cell = [35, 50, 35, 25, 90]

params2CellAlejandro = [20, 50, 15, 35, 140]

def getHoughCircles(img, paramss):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=paramss[0], 
                               param1=paramss[1], param2=paramss[2], 
                               minRadius=paramss[3], maxRadius=paramss[4])
          
    if circles is None:
        return circles
    if  circles.shape[1]>3:
        return circles[:,0:3,:]
    else:
         return circles
     
     
def drawCircles(img, circles):
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
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
        for aCirclePredIdx in range(circles.shape[1]):
            if aCirclePredIdx > 1:
                break;
            circleShape = Point(circles[0][aCirclePredIdx][0], circles[0][aCirclePredIdx][1]).buffer(circles[0][aCirclePredIdx][2])
            predsAsShape.append(circleShape)

    predsUnion = cascaded_union(predsAsShape)

    inter = predsUnion.intersection(gtUnion)
    union = predsUnion.union(gtUnion)
    jaccard = inter.area / union.area
    return jaccard
    

videos = ['E37', 'E40', 'E82', 'E56', 'E81', 'E99', 'E92', 'E66', 'E88',
    'E08', 'E45', 'E10', 'E32', 'E77', 'E42', 'E01', 'E57', 'E62',
    'E65', 'E14', 'E85', 'E84', 'E27', 'E94', 'E63', 'E02', 'E00',
    'E26', 'E78', 'E11', 'E50', 'E25', 'E13', 'E80', 'E03', 'E86',
    'E68', 'E19', 'E90', 'E06', 'E15', 'E18', 'E33', 'E69', 'E51',
    'E21', 'E41', 'E16', 'E48', 'E74', 'E30', 'E73', 'E46', 'E09',
    'E89', 'E76', 'E61', 'E28', 'E29', 'E04', 'E70', 'E38', 'E98',
    'E12', 'E75', 'E91', 'E95', 'E55', 'E47', 'E83']
    
avgBase = 0.0
avgOpt = 0.0

start = 75
end = 228



for aVideo in videos:
    avgBase=0
    avgOpt=0
    baseList=[]
    optimizedList=[]
    for frameIdx in range(start, end):
        gtCircles = getCircleLabelsForFrame(aVideo, frameIdx)

        imPath = os.path.join(baseDir, aVideo, 'Frame' + str(frameIdx).zfill(3) + '.png')
        img = loadAndPrepocessImage(imPath)

        circlesBase = getHoughCircles(img, bestGlobalParams)
        baseJaccard = calculateJaccard(circlesBase, gtCircles)
        baseList.append(baseJaccard)
        avgBase = avgBase + baseJaccard

        circlesOptimized = getHoughCircles(img, params2Cell)
        optimizedJaccard = calculateJaccard(circlesOptimized, gtCircles)
        optimizedList.append(optimizedJaccard)
        avgOpt = avgOpt + optimizedJaccard

        #print('Frame ', frameIdx, ' baseJaccard ', baseJaccard, ' optimizedJaccard ', optimizedJaccard)

    print('Video ',aVideo)
    print('avgBase ', avgBase / (end-start))
    print('avgOpt ', avgOpt / (end-start))
    
    baseAsArray=np.array(baseList)
    optimizedAsArray=np.array(optimizedList)
    
    print('baseAsArray.mean() ',baseAsArray.mean())
    print('optimizedAsArray.mean() ',optimizedAsArray.mean())

