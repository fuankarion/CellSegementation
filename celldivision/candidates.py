import cv2
import glob
import numpy as np
import os
import re
from scipy.ndimage import filters
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	
    
def loadVideoCube(videoPath):
    pathGT = os.path.join(videoPath, '_trajectories.txt')
    numLines=None
    with open(pathGT) as f:
       content = f.readlines()
       numLines=len(content)
       
       
    print('videoPath', videoPath)
    files = glob.glob(videoPath + '/*.png')
    
    files = sorted(files, key=natural_key)
    
    sampleImg = cv2.imread(os.path.join(videoPath, files[0]), 0)
    
    cube = np.zeros((sampleImg.shape[0], sampleImg.shape[1], numLines), np.int8)
    fileIdx = 0
    for aFile in files: 
        fileSourcePath = os.path.join(videoPath, aFile)
        img = cv2.imread(fileSourcePath, 0)
        cube[:, :, fileIdx] = img
        
        fileIdx = fileIdx + 1
        if fileIdx+1 >numLines:
            print('Break at ',fileIdx)
            break
        
    return cube

def getVoxelFromVideoCube(videoCube, startX, startY, startZ, size, timeSize):
    return  videoCube[startX:startX + size, startY:startY + size, startZ:startZ + timeSize]

def euclideanDistance(x1, x2):
    distance = np.subtract(x1, x2)
    return np.linalg.norm(distance) 
   
def getCubeLabel(centerCubeX, centerCubeY, centerCubeZ, boundaryTolerance, pathGT):
    pathGT = os.path.join(pathGT, '_trajectories.txt')
    with open(pathGT) as f:
        content = f.readlines()
        gtLine = content[centerCubeZ]
        #print(gtLine)
        
        tokens = gtLine.split('\t')
        #print(tokens)
        
        label = 0
        for tokensIdx in range(0, len(tokens)-3, 3):
            #print('Annotation: ', tokens[tokensIdx], ',', tokens[tokensIdx + 1], ',', tokens[tokensIdx + 2])
            if int(tokens[tokensIdx + 2]) > 0:#Cell candidate
                #print('Cell Annotation: ', tokens[tokensIdx], ',', tokens[tokensIdx + 1], ',', tokens[tokensIdx + 2])
                centroidAnnotation = np.array([int(tokens[tokensIdx]), int(tokens[tokensIdx + 1])])
                cubeCenter = np.array([centerCubeX, centerCubeY])
                dist = euclideanDistance(centroidAnnotation, cubeCenter)
                if dist < (int(tokens[tokensIdx + 2]) + boundaryTolerance):#Is inside
                    label = 1
                    
                    if dist > (int(tokens[tokensIdx + 2])-boundaryTolerance):
                        label = 2
                        
        return label