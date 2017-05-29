from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import sys
sys.path.append('../')
from utils import *
from svmUtil import *

datasetRoot = '/home/jcleon/Storage/ssd1/cellDivision/MouEmbTrkDtb'
numVideos = 100

###Optimal param
step = 20
voxelXY = 10
timeRange = 5
derivativeOrder = 4
kernelOpt = 'rbf'
COpt = 1000


dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)
featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelXY, step, timeRange, derivativeOrder, True)
featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelXY, step, timeRange, derivativeOrder, True)

featsTrain = preprocessing.scale(featsTrain)
featsTest = preprocessing.scale(featsTest)

baseSVM = svm.SVC(kernel=kernelOpt, C=COpt, class_weight='balanced')

print('Start SVM Train')
start = time.time()
baseSVM.fit(featsTrain, labelsTrain) 
end = time.time()
print('SVM Train Time ', end - start)

preds = baseSVM.predict(featsTest)
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labelsTest, preds, target_names=target_names)
f1Skore = f1_score(labelsTest, preds, average='weighted')
print(classificationReport)
            
