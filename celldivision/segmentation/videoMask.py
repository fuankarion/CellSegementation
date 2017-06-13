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


processPool = mp.Pool(20)

datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
videosTest = ['E01', 'E03', 'E11', 'E19', 'E21', 'E25', 'E26', 'E30', 'E31', 'E39', 'E40', 'E41', 'E45', 'E53', 'E57', 'E60', 'E61', 'E65', 'E66', 'E69', 'E70', 'E75', 'E81', 'E89', 'E90', 'E91', 'E92', 'E94', 'E98']

targetRoot = '/home/jcleon/Storage/ssd0/cellDivision/segmentations'
classifierDump = '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleCheat.pkl'
clf = joblib.load(classifierDump)

###Optimal param
voxelXY = 10
step = voxelXY
timeSize = 2
timeStep = 5
derivativeOrder = 4
tolerance = 0
includeCoordinate = True

dirFrames = os.path.join(datasetRoot, videosTest[0])
videoCube = loadVideoCube(dirFrames)

featCalculationArgs = []
zMax=90

for zSlice in range(0, zMax):
#for zSlice in range(0, videoCube.shape[2]):
    featCalculationArgs.append((videoCube, voxelXY, timeSize, step, timeStep, derivativeOrder, videosTest[0], datasetRoot, includeCoordinate, tolerance, zSlice))

descriptorsAndSpatialInfo = processPool.map(getTrainDataFromVideoSpatialInfo, featCalculationArgs)


for zSlice in range(0, zMax):
    print('Reconstruct slice ', zSlice)
    descriptors = descriptorsAndSpatialInfo[zSlice][0]
    labels = descriptorsAndSpatialInfo[zSlice][1]
    spatialInfo = descriptorsAndSpatialInfo[zSlice][2]
    #print('Final ', descriptors.shape)
    descriptors = preprocessing.scale(descriptors)

    #print('Start predictions for', videosTest[0])
    preds = clf.predict(descriptors)
    target_names = ['Background', 'Cell', 'Boundary']
    classificationReport = classification_report(labels, preds, target_names=target_names)
    print(classificationReport)

    #Write masks
    targetDir = os.path.join(targetRoot, videosTest[0])
    segmentationMask = np.zeros((videoCube.shape[0], videoCube.shape[1], 1))

    #print('len(spatialInfo) ', len(spatialInfo))
    for dataIdx in range(0, len(spatialInfo)):

        aVoxelCoordinate = spatialInfo[dataIdx]
        #print('aVoxelCoordinate',aVoxelCoordinate)

        segmentationMask[aVoxelCoordinate[0]-(step / 2):aVoxelCoordinate[0] + (step / 2),
        aVoxelCoordinate[1]-(step / 2):aVoxelCoordinate[1] + (step / 2)] = int(preds[dataIdx] * 255)

    segmentationMask=np.concatenate((segmentationMask, segmentationMask), axis=2)
    segmentationMask=np.concatenate((segmentationMask, segmentationMask), axis=2)
    imgPath = os.path.join(targetDir, str(zSlice) + '.jpg')
    scipy.misc.imsave(imgPath, segmentationMask)

