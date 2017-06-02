import os
from sklearn import preprocessing
import sys
sys.path.append('../')
from utils import *
sys.path.append('../svm')
from svmUtil import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report

datasetRoot = '/home/jcleon/Storage/ssd1/cellDivision/MouEmbTrkDtb'
videosTest = ['E00', 'E03', 'E04', 'E10', 'E13', 'E15', 'E32', 'E33', 'E34', 
'E36', 'E37', 'E38', 'E39', 'E48', 'E52', 'E64', 'E66', 'E68', 'E70', 'E76', 
'E80', 'E81', 'E87', 'E89', 'E91', 'E94', 'E95', 'E96', 'E99']

classifierDump = '/home/jcleon/Storage/ssd1/cellDivision/models/svmEnsembleCheat.pkl'
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

preds = clf.predict(descriptors)
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labels, preds, target_names=target_names)
print(classificationReport)

#Write masks
for aVoxelCoordinate in spatialInfo:
    
#featsTest = preprocessing.scale(featsTest)


