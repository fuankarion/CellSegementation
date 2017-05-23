import cv2
import glob
import numpy as np
import os
import random
import re
from scipy.ndimage import filters

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
        result[:, :, :, 0 + (i * 8)] = filters.gaussian_filter(data, [1, 1, 1], order=i)   
        result[:, :, :, 1 + (i * 8)] = filters.gaussian_filter(data, [1, 1, -1], order=i)   
        result[:, :, :, 2 + (i * 8)] = filters.gaussian_filter(data, [1, -1, 1], order=i)   
        result[:, :, :, 3 + (i * 8)] = filters.gaussian_filter(data, [1, -1, -1], order=i)   

        result[:, :, :, 4 + (i * 8)] = filters.gaussian_filter(data, [-1, 1, 1], order=i)   
        result[:, :, :, 5 + (i * 8)] = filters.gaussian_filter(data, [-1, 1, -1], order=i)   
        result[:, :, :, 6 + (i * 8)] = filters.gaussian_filter(data, [-1, -1, 1], order=i)   
        result[:, :, :, 7 + (i * 8)] = filters.gaussian_filter(data, [-1, -1, -1], order=i)   
    
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
        
    return cube

def getVoxelFromVideoCube(videoCube, startX, startY, startZ, size, timeSize):
    return  videoCube[startX:startX + size, startY:startY + size, startZ:startZ + timeSize]
    
   
def getCubeLabel(centerCubeX, centerCubeY, centerCubeZ, boundaryTolerance, pathGT):
    pathGT = os.path.join(pathGT, '_trajectories.txt')
    with open(pathGT) as f:
        content = f.readlines()
        gtLine = content[centerCubeZ]
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
    
def euclidenaDistance(x1, x2):
    distance = np.subtract(x1, x2)
    return np.linalg.norm(distance)

def getTrainDataFromVideo(videoCube, voxelSize, step, timeSize, order, sequenceName):
    dirFrames = os.path.join(datasetRoot, sequenceName)
    descriptors = np.zeros((1, 8 * order))
    labels = np.zeros(1)
    for x in range(0, videoCube.shape[0]-voxelSize, step):
        for y in range(0, videoCube.shape[1]-voxelSize, step):
            for z in range(0, videoCube.shape[2]-timeSize, step):
                voxelLabel = getCubeLabel(x, y, z, 5, dirFrames)

                if voxelLabel == 0:
                    ignoreFlag = random.uniform(0.0, 1.0)
                    if ignoreFlag <= 0.8:
                        continue

                aVoxel = getVoxelFromVideoCube(videoCube, x, y, z, voxelSize, timeSize)
                voxelDescriptor = getSTIPDescriptor(aVoxel, order)

                #print('voxelDescriptor.shape ',voxelDescriptor.shape)
                descriptors = np.concatenate((descriptors, voxelDescriptor), axis=0)
                labels = np.concatenate((labels, np.array([voxelLabel])), axis=0)
    return descriptors, labels
    
