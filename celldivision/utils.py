import cv2
import glob
import numpy as np
import os
import random
import re
from scipy.ndimage import filters
import time

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def getSTIPDescriptor(data, order):
    result = np.zeros((data.shape[0], data.shape[1], data.shape[2], 8 * order))
    decriptor = []
    centerX = int(result.shape[0] / 2)
    centerY = int(result.shape[1] / 2)
    centerZ = int(result.shape[2] / 2)
    
    for i in range (0, order):
        result[:,:,:, 0 + (i * 8)] = filters.gaussian_filter(data, [1, 1, 1], order=i)   
        result[:,:,:, 1 + (i * 8)] = filters.gaussian_filter(data, [1, 1, -1], order=i)   
        result[:,:,:, 2 + (i * 8)] = filters.gaussian_filter(data, [1, -1, 1], order=i)   
        result[:,:,:, 3 + (i * 8)] = filters.gaussian_filter(data, [1, -1, -1], order=i)   

        result[:,:,:, 4 + (i * 8)] = filters.gaussian_filter(data, [-1, 1, 1], order=i)   
        result[:,:,:, 5 + (i * 8)] = filters.gaussian_filter(data, [-1, 1, -1], order=i)   
        result[:,:,:, 6 + (i * 8)] = filters.gaussian_filter(data, [-1, -1, 1], order=i)   
        result[:,:,:, 7 + (i * 8)] = filters.gaussian_filter(data, [-1, -1, -1], order=i)   
    
    for i in range (0, order):
        decriptor.append(result[centerX, centerY, centerZ, 0 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 1 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 2 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 3 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 4 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 5 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 6 + (i * 8)])
        decriptor.append(result[centerX, centerY, centerZ, 7 + (i * 8)])
    
    return np.expand_dims(np.array(decriptor), axis=0)

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
        cube[:,:, fileIdx] = img
        
        fileIdx = fileIdx + 1
        if fileIdx + 1 > numLines:
            print('Annotations for ', str(fileIdx) + ' images')
            break
    end = time.time()
    print('Load cube time ', end - start)
    return cube

def getVoxelFromVideoCube(videoCube, startX, startY, startZ, size, timeSize):
    return  videoCube[startX:startX + size, startY:startY + size, startZ:startZ + timeSize]
   
def getCubeLabel(centerCubeX, centerCubeY, centerCubeZ, boundaryTolerance, contentGT):
    gtLine = contentGT[centerCubeZ]
    tokens = gtLine.split('\t')
    label = 0

    for tokensIdx in range(0, len(tokens)-3, 3):
        if int(tokens[tokensIdx + 2]) > 0:#Cell candidate
            centroidAnnotation = np.array([int(tokens[tokensIdx]), int(tokens[tokensIdx + 1])])
            cubeCenter = np.array([centerCubeX, centerCubeY])
            dist = euclidenaDistance(centroidAnnotation, cubeCenter)
            if dist < (int(tokens[tokensIdx + 2]) + boundaryTolerance):#Is inside
                label = 1
                if dist > (int(tokens[tokensIdx + 2])-boundaryTolerance):
                    label = 2
                        
    return label

def getFrameStageLabel(datasetRoot, videoId, frameIdx):
    pathGT = os.path.join(datasetRoot, videoId, '_trajectories.txt')
    with open(pathGT) as f:
        content = f.readlines()
    
    targetLine = content[frameIdx-1]
    tokens = targetLine.split('\t')
    
    if int(tokens[2]) != 0:
        return 1
    elif int(tokens[5]) != 0 and int (tokens[8]) != 0:
        return 2
    elif int(tokens[11]) != 0 and int (tokens[14]) != 0 and int (tokens[17]) != 0:
        return 3
    else:
        return 0
    
def euclidenaDistance(x1, x2):
    distance = np.subtract(x1, x2)
    return np.linalg.norm(distance)
  
def addXYCoordinatesToDescriptor(voxelDescriptor, x, y, videoCube):
    normalizedCoordinates = [(float(x) / float(videoCube.shape[0])), (float(y) / float(videoCube.shape[1]))]
    coordinateArray = np.array(normalizedCoordinates)
    coordinateArray = np.transpose(np.expand_dims(coordinateArray, axis=1))
    voxelDescriptor = np.concatenate((voxelDescriptor, coordinateArray), axis=1)
    return voxelDescriptor

#Pararlel code, tuple is a simple hack    
def getTrainDataFromVideo(tupleArgs):
    start = time.time()
    videoCube = tupleArgs[0]
    voxelSize = tupleArgs[1]
    step = tupleArgs[2]
    timeSize = tupleArgs[3]
    order = tupleArgs[4]
    sequenceName = tupleArgs[5]
    datasetRoot = tupleArgs[6]
    includeCoordinates = tupleArgs[7]
    tolerance = tupleArgs[8]
    includeBackground = tupleArgs[9]
        
    print('Process Feats from ', sequenceName)
    dirFrames = os.path.join(datasetRoot, sequenceName)
    if includeCoordinates:
        descriptors = np.zeros((1, (8 * order) + 2))
    else:
        descriptors = np.zeros((1, 8 * order))
        
    labels = np.zeros(1)
    
    contentGT = None
    pathGT = os.path.join(dirFrames, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
        
    for x in range(0, videoCube.shape[0]-voxelSize, step):
        for y in range(0, videoCube.shape[1]-voxelSize, step):
            for z in range(0, videoCube.shape[2]-timeSize, step):
                voxelLabel = getCubeLabel(x, y, z, tolerance, contentGT)

                
                if voxelLabel == 0:
                    if includeBackground:
                        ignoreFlag = random.uniform(0.0, 1.0)
                        if ignoreFlag <= 0.8:
                            continue
                    else:
                        continue                           

                aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
                voxelDescriptor = getSTIPDescriptor(aVoxel, order)
                if includeCoordinates:
                    voxelDescriptor = addXYCoordinatesToDescriptor(voxelDescriptor, x, y, videoCube)

                descriptors = np.concatenate((descriptors, voxelDescriptor), axis=0)
                labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)
    end = time.time()
    descriptors = np.delete(descriptors, 0, 0)
    labels = np.delete(labels, 0, 0)
    print('Process Feats Time ', end - start)
    return (descriptors, labels)
    

def getTrainDataFromVideoSpatialInfo(tupleArgs):
    start = time.time()
    videoCube = tupleArgs[0]
    voxelSize = tupleArgs[1]
    step = tupleArgs[2]
    timeSize = tupleArgs[3]
    order = tupleArgs[4]
    sequenceName = tupleArgs[5]
    datasetRoot = tupleArgs[6]
    includeCoordinates = True
    tolerance = tupleArgs[7]
    includeBackground = tupleArgs[8]
    
    print('Process Feats from ', sequenceName)
    dirFrames = os.path.join(datasetRoot, sequenceName)
    if includeCoordinates:
        descriptors = np.zeros((1, (8 * order) + 2))
    else:
        descriptors = np.zeros((1, 8 * order))
        
    labels = np.zeros(1)
    spatialInfo = []
    
    contentGT = None
    pathGT = os.path.join(dirFrames, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
        
    for x in range(0, videoCube.shape[0]-voxelSize, step):
        for y in range(0, videoCube.shape[1]-voxelSize, step):
            for z in range(0, videoCube.shape[2]-timeSize, step):
                voxelLabel = getCubeLabel(x, y, z, 0, contentGT)

                if voxelLabel == 0:
                    if not includeBackground:
                        continue
   

                aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
                voxelDescriptor = getSTIPDescriptor(aVoxel, order)
                if includeCoordinates:
                    voxelDescriptor = addXYCoordinatesToDescriptor(voxelDescriptor, x, y, videoCube)

                descriptors = np.concatenate((descriptors, voxelDescriptor), axis=0)
                labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)
                spatialInfo.append((x, y, z))
    end = time.time()
    print('Process Feats Time ', end - start)
    return (descriptors, labels, spatialInfo)
    
"""
label = getFrameStageLabel('/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb/', 'E90', 228)
print('Label ', label)
"""
