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

datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
videosTest =  ['E01', 'E03', 'E11', 'E19', 'E21', 'E25', 'E26', 'E30', 'E31', 'E39', 'E40', 'E41', 'E45', 'E53', 'E57', 'E60', 'E61', 'E65', 'E66', 'E69', 'E70', 'E75', 'E81', 'E89', 'E90', 'E91', 'E92', 'E94', 'E98']

targetRoot = '/home/jcleon/Storage/ssd0/cellDivision/segmentations'
classifierDump = '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleCheat.pkl'
clf = joblib.load(classifierDump)

###Optimal param
voxelXY = 10
step = voxelXY
timeRange = 5
derivativeOrder = 4
tolerance = 0

dirFrames = os.path.join(datasetRoot, videosTest[0])
videoCube = loadVideoCube(dirFrames)

featCalculationArgs = (videoCube, voxelXY, step, timeRange, derivativeOrder, videosTest[0], datasetRoot, tolerance, True)
descriptorsAndSpatialInfo = getTrainDataFromVideoSpatialInfo(featCalculationArgs)

descriptors = descriptorsAndSpatialInfo[0] 
labels = descriptorsAndSpatialInfo[1] 
spatialInfo = descriptorsAndSpatialInfo[2]
descriptors = preprocessing.scale(descriptors)

print('Start predictions for',videosTest[0])
preds = clf.predict(descriptors)
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labels, preds, target_names=target_names)
print(classificationReport)

#Write masks
targetDir = os.path.join(targetRoot, videosTest[0])
segmentationMask = np.zeros((videoCube.shape))
for dataIdx in range(0, len(spatialInfo)):
    
    aVoxelCoordinate = spatialInfo[dataIdx]
    
    segmentationMask[aVoxelCoordinate[0]-(step / 2):aVoxelCoordinate[0] + (step / 2),
    aVoxelCoordinate[1]-(step / 2):aVoxelCoordinate[1] + (step / 2), 
    aVoxelCoordinate[2]-(timeRange / 2):aVoxelCoordinate[2] + (timeRange / 2)] = int(preds[dataIdx] * 255)
    
    #print('[aVoxelCoordinate[0]-(voxelXY / 2):aVoxelCoordinate[0] + (voxelXY / 2)', aVoxelCoordinate[0]-(voxelXY / 2), aVoxelCoordinate[0] + (voxelXY / 2))
    #print('[aVoxelCoordinate[1]-(voxelXY / 2):aVoxelCoordinate[1] + (voxelXY / 2)', aVoxelCoordinate[1]-(voxelXY / 2), aVoxelCoordinate[1] + (voxelXY / 2))
    #print('[aVoxelCoordinate[2]-(voxelXY / 2):aVoxelCoordinate[2] + (voxelXY / 2)', aVoxelCoordinate[2]-(timeRange / 2), aVoxelCoordinate[2] + (timeRange / 2))
   
    #print('reds[dataIdx] * 255 ',preds[dataIdx] * 255)
        
imageIdx = 0
for imageSliceIdx in range(0, segmentationMask.shape[2]):
    imgPath = os.path.join(targetDir, str(imageSliceIdx) + '.png')
    scipy.misc.imsave(imgPath, segmentationMask[:,:, imageSliceIdx])
    
    
#featsTest = preprocessing.scale(featsTest)


