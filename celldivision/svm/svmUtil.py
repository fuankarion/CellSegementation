import multiprocessing as mp
import os
import random
from random import shuffle
import re
import sys
sys.path.append('../')
from utils import *

processPool = mp.Pool(30)

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def createTrainAndTestSubSets(datasetRoot, numVideos):
    dirsTrain = []
    dirsTest = []
    
    allDirs = []
    
    videoDirs = os.listdir(datasetRoot)
    shuffle(videoDirs)

    for aVideoDir in videoDirs:
        allDirs.append(aVideoDir)
        if len(allDirs)  > numVideos:
            break
    shuffle(allDirs)
    
    dirsTrain = allDirs[:int(numVideos * 0.7)]
    dirsTest = allDirs[int(numVideos * 0.7) + 1:]
            
    dirsTrain = sorted(dirsTrain, key=natural_key)
    dirsTest = sorted(dirsTest, key=natural_key)
    print('dirsTrain ', dirsTrain)
    print('dirsTest ', dirsTest)
    return dirsTrain, dirsTest

def createEvaluationFold(datasetRoot):
    numVideos = 100
    dirsTrain = []
    dirsVal = []
    dirsTest = []
    
    allDirs = []
    
    videoDirs = os.listdir(datasetRoot)
    shuffle(videoDirs)

    for aVideoDir in videoDirs:
        allDirs.append(aVideoDir)
        if len(allDirs)  > numVideos:
            break
    shuffle(allDirs)
    
    dirsTrain = allDirs[:int(numVideos * 0.5)]
    remainig = allDirs[int(numVideos * 0.5) + 1:]
    
    dirsVal = remainig[:int(numVideos * 0.5)]
    dirsTest = remainig[int(numVideos * 0.5) + 1:]
            
    dirsTrain = sorted(dirsTrain, key=natural_key)
    dirsVal = sorted(dirsVal, key=natural_key)
    dirsTest = sorted(dirsTest, key=natural_key)
    print('dirsTrain ', dirsTrain)
    print('dirsVal ', dirsVal)
    print('dirsTest ', dirsTest)
    return dirsTrain, dirsVal, dirsTest

 
def loadSetFromVideos(videoDirs, datasetRoot, voxelSize, timeSize, step, timeStep, order, includeCoordinate, tolerance, includeBackground):
    featsSet = None
    labelsSet = None

    featCalculationArgs = []
    for aVideoDir in videoDirs:
        dirFrames = os.path.join(datasetRoot, aVideoDir)
        videoCube = loadVideoCube(dirFrames)
        featCalculationArgs.append((videoCube, voxelSize, timeSize, step, timeStep, order, aVideoDir, datasetRoot, includeCoordinate, tolerance, includeBackground))
    
    print('process feat data')
    data = processPool.map(getTrainDataFromVideo, featCalculationArgs)
    
    print('Unroll feat data')
    featsSet = None
    labelsSet = None
    for aCubeResult in data:
        if featsSet == None:
            featsSet = aCubeResult[0]
            labelsSet = aCubeResult[1]
        else:
            featsSet = np.concatenate((featsSet, aCubeResult[0]), axis=0)
            labelsSet = np.concatenate((labelsSet, aCubeResult[1]), axis=0)
    return featsSet, labelsSet 


def fullSetData(videoDirs, datasetRoot, voxelSize, step, timeSize, order, includeCoordinate):
    featsSet = None
    labelsSet = None

    featCalculationArgs = []
    for aVideoDir in videoDirs:
        dirFrames = os.path.join(datasetRoot, aVideoDir)
        videoCube = loadVideoCube(dirFrames)
        
        featCalculationArgs.append((videoCube, voxelSize, step, timeSize, order, aVideoDir, datasetRoot, includeCoordinate))
    
    print('process feat data')
    data = processPool.map(fullSetData, featCalculationArgs)
    
    print('Unroll feat data')
    featsSet = None
    labelsSet = None
    for aCubeResult in data:
        if featsSet == None:
            featsSet = aCubeResult[0]
            labelsSet = aCubeResult[1]
        else:
            featsSet = np.concatenate((featsSet, aCubeResult[0]), axis=0)
            labelsSet = np.concatenate((labelsSet, aCubeResult[1]), axis=0)
    return featsSet, labelsSet 
