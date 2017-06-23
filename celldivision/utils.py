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
    pathGT = os.path.join(videoPath, '_trajectories.txt')
    numLines=None
    with open(pathGT) as f:
       content = f.readlines()
       numLines=len(content)
       
       
    print('videoPath', videoPath)
    files = glob.glob(videoPath + '/*.png')
    
    files = sorted(files, key=natural_key)
    

    sampleImg_ = cv2.imread(os.path.join(videoPath, files[0]), 0)
    
    x = np.linspace(1,255,sampleImg_.shape[0])    
    y = np.linspace(1,255,sampleImg_.shape[1])        
    distx, disty = np.meshgrid(x,y)

    #distx = np.around((distx-distx.min())/(distx.max()-distx.min()))*255
    #disty = np.around((disty-disty.min())/(disty.max()-disty.min()))*255
    
    sampleImg = np.zeros((sampleImg_.shape[0],sampleImg_.shape[1],3))

    cube = np.zeros((sampleImg.shape[0], sampleImg.shape[1],sampleImg.shape[2], numLines), np.uint8)
    fileIdx = 0
    for aFile in files: 
        fileSourcePath = os.path.join(videoPath, aFile)
        img = cv2.imread(fileSourcePath, 0)
        cube[:, :, 0,fileIdx] = img
        cube[:, :, 1,fileIdx] = distx
        cube[:, :, 2,fileIdx] = disty
      
        fileIdx = fileIdx + 1
        if fileIdx+1 > numLines:
            print('Break at ',fileIdx)
            break
        
    return cube

def getVoxelFromVideoCube(videoCube, startX, startY, startZ, size, timeSize):
    return  videoCube[startX:startX + size, startY:startY + size,:, startZ:startZ + timeSize]
   
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
            for z in range(0, videoCube.shape[3]-timeSize, step):
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

def getTrainDataFromVideoSpatialInfo(tupleArgs):
    videoCube = tupleArgs[0]
    voxelSize = tupleArgs[1]
    timeSize = tupleArgs[2]
    step = tupleArgs[3]
    timeStep = tupleArgs[4]
    order = tupleArgs[5]
    sequenceName = tupleArgs[6]
    datasetRoot = tupleArgs[7]
    includeCoordinates = [8]
    tolerance = tupleArgs[9]
    z = tupleArgs[10]
    
    #print('Set up Memory')
    dirFrames = os.path.join(datasetRoot, sequenceName)

    descriptors = []
    labels = []
    spatialInfo = []
    
    print('Load GT')
    contentGT = None
    pathGT = os.path.join(dirFrames, '_trajectories.txt')
    with open(pathGT) as f:
        contentGT = f.readlines()
        
    print('Process Feats from ', sequenceName, ' Slice ', z)
    start = time.time()    
    for x in range(int(voxelSize / 2), videoCube.shape[0]-int(voxelSize / 2), step):
        #print('sequenceName ', sequenceName, ' x ', x)
        for y in range(int(voxelSize / 2), videoCube.shape[1]-int(voxelSize / 2), step):
           
            voxelLabel = getVoxelLabel(x, y, z, 0, contentGT)

            aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
            #voxelDescriptor = getSTIPDescriptor(aVoxel, order)
            #if includeCoordinates:
            #    voxelDescriptor = addXYCoordinatesToDescriptor(voxelDescriptor, x, y, videoCube)

            #print('voxelDescriptor.shape', voxelDescriptor.shape)
            descriptors.append(aVoxel)
            labels.append(voxelLabel)
            spatialInfo.append((x, y, z))
    end = time.time()
    print('Process Feats Time ', end - start)
    
    start = time.time()    
    labels = np.squeeze(np.array(labels))
    descriptors = np.squeeze(np.array(descriptors))
    end = time.time()
    #print('NP Array Conversion Time ', end - start)
    #print('descriptors.shape', descriptors.shape)
    return (descriptors, labels, spatialInfo)