import cv2
import numpy as np
import os
from shapely.geometry import Point
from shapely.ops import cascaded_union

baseDir = '/home/fuanka/Downloads/MouEmbTrkDtb/MouEmbTrkDtb'

bestGlobalParams = [35, 50, 35, 70, 140]

params1Cell = [35, 50, 30, 90, 140]
params2Cell = [35, 50, 20, 50, 90]
params3Cell = [35, 50, 30, 25, 70]

def getHoughCircles(img, paramss):
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=paramss[0], 
                               param1=paramss[1], param2=paramss[2], 
                               minRadius=paramss[3], maxRadius=paramss[4])
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
    
def calculateJaccard(circles,gtCircles):
    #Better jaccard Calculation
    gtAsShape = []
    for aCircleGTIdx in range(len(gtCircles)):
        circleShape = Point(gtCircles[aCircleGTIdx][0], gtCircles[aCircleGTIdx][1]).buffer(gtCircles[aCircleGTIdx][2])
        gtAsShape.append(circleShape)
    gtUnion = cascaded_union(gtAsShape)

    predsAsShape = []
    for aCirclePredIdx in range(circles.shape[1]):
        circleShape = Point(circles[0][aCirclePredIdx][0], circles[0][aCirclePredIdx][1]).buffer(circles[0][aCirclePredIdx][2])
        predsAsShape.append(circleShape)
    predsUnion = cascaded_union(predsAsShape)

    inter = predsUnion.intersection(gtUnion)
    union = predsUnion.union(gtUnion)
    jaccard = inter.area / union.area
    print('Jaccard', jaccard)
    

video = 'E10'
frameIdx = 301
imPath = os.path.join(baseDir, video, 'Frame' + str(frameIdx).zfill(3) + '.png')

gtCircles = getCircleLabelsForFrame(video, frameIdx)
img = loadAndPrepocessImage(imPath)
circles = getHoughCircles(img, params3Cell)#bestGlobalParams #params3Cell

#drawCircles(img, circles)

calculateJaccard(circles,gtCircles)


