import os
from sklearn import preprocessing
import sys
sys.path.append('../')
from utils import *
sys.path.append('../svm')
from svmUtil import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report
import scipy.misc

processPool = mp.Pool(40)

def reconstructSlice(rargs):
    zSlice = rargs[0]
    descriptorsAndSpatialInfo = rargs[1]
    clf = rargs[2]
    targetRoot = rargs[3]
    sequenceName = rargs[4]
    videoShape = rargs[5]    
    
    print('Reconstruct slice ', zSlice)
    descriptors = descriptorsAndSpatialInfo[0]
    labels = descriptorsAndSpatialInfo[1]
    spatialInfo = descriptorsAndSpatialInfo[2]

    #NOT FOR NN
    descriptors = preprocessing.scale(descriptors)

    preds = clf.predict(descriptors)
    #target_names = ['Background', 'Cell', 'Boundary']
    #classificationReport = classification_report(labels, preds, target_names=target_names)
    #print(classificationReport)

    #Write masks
    targetDir = os.path.join(targetRoot, sequenceName)
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)
        print ('created ', targetDir)
        
    segmentationMask = np.zeros((videoShape[0], videoShape[1], 1))

    for dataIdx in range(0, len(spatialInfo)):
        aVoxelCoordinate = spatialInfo[dataIdx]

        segmentationMask[aVoxelCoordinate[0]-(step / 2):aVoxelCoordinate[0] + (step / 2),
        aVoxelCoordinate[1]-(step / 2):aVoxelCoordinate[1] + (step / 2)] = int(preds[dataIdx] * 255)

    segmentationMask = np.concatenate((segmentationMask, segmentationMask), axis=2)
    segmentationMask = np.concatenate((segmentationMask, segmentationMask), axis=2)
    imgPath = os.path.join(targetDir, str(zSlice) + '.jpg')
    scipy.misc.imsave(imgPath, segmentationMask)

datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
#videosTest = ['E01', 'E03', 'E11', 'E19', 'E21', 'E25', 'E26', 'E30', 'E31', 'E39', 'E40', 'E41', 'E45', 'E53', 'E57', 'E60', 'E61', 'E65', 'E66', 'E69', 'E70', 'E75', 'E81', 'E89', 'E90', 'E91', 'E92', 'E94', 'E98']
videosTest = ['E03', 'E11', 'E19', 'E21', 'E25', 'E26', 'E30', 'E31', 'E39', 'E40', 'E41', 'E45', 'E53', 'E57', 'E60', 'E61', 'E65', 'E66', 'E69', 'E70', 'E75', 'E81', 'E89', 'E90', 'E91', 'E92', 'E94', 'E98']

targetRoot = '/home/jcleon/Storage/ssd0/cellDivision/segmentations'
classifierDump = '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleCheat.pkl'
clf = joblib.load(classifierDump)

###Optimal param
voxelXY = 10
step = voxelXY
timeSize = 5
timeStep = 5
derivativeOrder = 4
tolerance = 0
includeCoordinate = True


for aVideo in videosTest:
    dirFrames = os.path.join(datasetRoot, aVideo)
    videoCube = loadVideoCube(dirFrames)

    zMax = videoCube.shape[2]

    featCalculationArgs = []
    for zSlice in range(0, zMax):
        #Get features
        featCalculationArgs.append((videoCube, voxelXY, timeSize, step, timeStep, derivativeOrder, aVideo, datasetRoot, includeCoordinate, tolerance, zSlice))

    descriptorsAndSpatialInfo = processPool.map(getTrainDataFromVideoSpatialInfo, featCalculationArgs)

    for zSlice in range(0, zMax):
        maskCalculationArgs = (zSlice, descriptorsAndSpatialInfo[zSlice], clf, targetRoot, aVideo, videoCube.shape)
        reconstructSlice(maskCalculationArgs)

