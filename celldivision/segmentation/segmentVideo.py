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
videoId = ['E00', 'E02', 'E07']

classifierDump = '/home/jcleon/Storage/ssd1/cellDivision/models/svmEnsembleCheat.pkl'
clf = joblib.load(classifierDump)


###Optimal param
voxelXY = 10
step = voxelXY
timeRange = 5
derivativeOrder = 4
tolerance = 0

dirFrames = os.path.join(datasetRoot, videoId[0])
videoCube = loadVideoCube(dirFrames)

featCalculationArgs = (videoCube, voxelXY, step, timeRange, derivativeOrder, videoId[0], datasetRoot, True)
descriptorsAndSpatialInfo = getTrainDataFromVideoSpatialInfo(featCalculationArgs)

descriptors = descriptorsAndSpatialInfo[0] 
labels = descriptorsAndSpatialInfo[1] 
spatialInfo = descriptorsAndSpatialInfo[2]
descriptors = preprocessing.scale(descriptors)

preds = clf.predict(descriptors)
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labels, preds, target_names=target_names)
print(classificationReport)
#featsTest = preprocessing.scale(featsTest)


