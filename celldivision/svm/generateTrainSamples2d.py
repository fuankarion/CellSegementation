datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
numVideos = 100

import cv2
import glob
import numpy as np
import os
import random
from random import shuffle
import re
import time
from random import randint


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def loadVideoCube(videoPath):
    start = time.time()
    pathGT = os.path.join(videoPath, '_trajectories.txt')
    numLines = None
    with open(pathGT) as f:
        content = f.readlines()
        numLines = len(content)
       
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
        if fileIdx + 1 > numLines:
            print('Annotations for ', str(fileIdx) + ' images')
            break
    end = time.time()
    print('Load cube time ', end - start)
    return cube

def getVoxelLabel(centerCubeX, centerCubeY, centerCubeZ, boundaryTolerance, contentGT):
    gtLine = contentGT[centerCubeZ]
    tokens = gtLine.split('\t')
    label = 0
 
    for tokensIdx in range(0, len(tokens)-3, 3):
        cellRadius = int(tokens[tokensIdx + 2])
        if  cellRadius > 0:#Cell candidate
            centroidAnnotation = np.array([int(tokens[tokensIdx]), int(tokens[tokensIdx + 1])])
            cubeCenter = np.array([centerCubeY, centerCubeX])
            dist = euclidenaDistance(centroidAnnotation, cubeCenter)       
            
            if dist < (cellRadius + boundaryTolerance):#Is inside
                label = 1
                if dist > (cellRadius -boundaryTolerance):
                    return 2
                   
    return label

def euclidenaDistance(x1, x2):
    distance = np.subtract(x1, x2)
    return np.linalg.norm(distance)

def getVoxelFromVideoCube(videoCube, startX, startY, startZ, size, timeSize):
    halfSize = int(size / 2)
    return  videoCube[(startY-halfSize):(startY + halfSize), (startX-halfSize):(startX + halfSize), startZ:(startZ + timeSize)]


dirsTrain = []
dirsTest = []
allDirs = []

targetRoot = '/home/jcleon/Storage/disk0/CellVsBoundary'

videoDirs = os.listdir(datasetRoot)
shuffle(videoDirs)

for aVideoDir in videoDirs:
    allDirs.append(aVideoDir)
    if len(allDirs)  > numVideos:
        break
shuffle(allDirs)

dirsTrain = allDirs[:int(numVideos * 0.7)]
dirsTest = allDirs[int(numVideos * 0.7) + 1:]

voxelSize = 50
step = 40
timeStep=4
timeSize = 1
tolerance = 4

for aVideoDir in dirsTrain[:10]:
    dirFrames = os.path.join(datasetRoot, aVideoDir)
    targetPath = os.path.join(targetRoot, 'train')
    videoCube = loadVideoCube(dirFrames)
    
    contentGT = None
    pathGT = os.path.join(dirFrames, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
        
    for x in range(int(voxelSize / 2), videoCube.shape[0]-int(voxelSize / 2),  randint(step/2, step)):
    #for x in range(240, videoCube.shape[0]-int(voxelSize / 2), step):
        for y in range(int(voxelSize / 2), videoCube.shape[1]-int(voxelSize / 2),  randint(step/2, step)):
        #for y in range(240, videoCube.shape[1]-int(voxelSize / 2), step):
            for z in range(int(timeSize / 2), videoCube.shape[2]-int(timeSize / 2), randint(timeStep/2, timeStep)):
            #for z in range(0, 1):
                #print('aVideoDir ', aVideoDir)
                voxelLabel = getVoxelLabel(x, y, z, tolerance, contentGT)
                
                if voxelLabel == 0:
                    continue
                    #ignoreFlag = random.uniform(0.0, 1.0)
                    #if ignoreFlag <= 0.98:
                    
                
                if voxelLabel == 1:
                    
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.55:
                        continue
                
                
                aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
                
                #print('aVoxel.shape ',aVoxel.shape)
                #print('voxelLabel ',voxelLabel)
                
                if voxelLabel == 0:
                    finalTargetPah = os.path.join(targetPath, 'background')
                elif voxelLabel == 1:
                    finalTargetPah = os.path.join(targetPath, 'cell')
                elif voxelLabel == 2:
                    finalTargetPah = os.path.join(targetPath, 'boundary')
                    
                finalTargetPah = os.path.join(finalTargetPah, 'x' + str(x) + 'y' + str(y) + 'z' + str(z) + '.bmp')
                
                print('write', finalTargetPah)
                print('aVideoDir ', aVideoDir)
                cv2.imwrite(finalTargetPah, aVoxel)
                
                
for aVideoDir in dirsTest[10:15]:
    dirFrames = os.path.join(datasetRoot, aVideoDir)
    targetPath = os.path.join(targetRoot, 'test')
    videoCube = loadVideoCube(dirFrames)
    
    contentGT = None
    pathGT = os.path.join(dirFrames, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
        
    for x in range(int(voxelSize / 2), videoCube.shape[0]-int(voxelSize / 2), randint(step/2, step)):
    #for x in range(240, videoCube.shape[0]-int(voxelSize / 2), step):
        for y in range(int(voxelSize / 2), videoCube.shape[1]-int(voxelSize / 2), randint(step/2, step)):
        #for y in range(240, videoCube.shape[1]-int(voxelSize / 2), step):
            for z in range(int(timeSize / 2), videoCube.shape[2]-int(timeSize / 2), randint(timeStep/2, timeStep)):
            #for z in range(0, 1):
                #print('aVideoDir ', aVideoDir)
                voxelLabel = getVoxelLabel(x, y, z, tolerance, contentGT)
                
                if voxelLabel == 0:
                    continue
                    #ignoreFlag = random.uniform(0.0, 1.0)
                    #if ignoreFlag <= 0.98:
                    
                
                if voxelLabel == 1:
                    
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.65:
                        continue
                
                
                aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
                
                #print('aVoxel.shape ',aVoxel.shape)
                #print('voxelLabel ',voxelLabel)
                
                if voxelLabel == 0:
                    finalTargetPah = os.path.join(targetPath, 'background')
                elif voxelLabel == 1:
                    finalTargetPah = os.path.join(targetPath, 'cell')
                elif voxelLabel == 2:
                    finalTargetPah = os.path.join(targetPath, 'boundary')
                    
                finalTargetPah = os.path.join(finalTargetPah, 'x' + str(x) + 'y' + str(y) + 'z' + str(z) + '.bmp')
                
                print('write', finalTargetPah)
                print('aVideoDir ', aVideoDir)
                cv2.imwrite(finalTargetPah, aVoxel)