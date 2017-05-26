
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import sys
sys.path.append('../')
from utils import *
from svmUtil import *

datasetRoot = '/home/jcleon/Storage/ssd1/cellDivision/MouEmbTrkDtb'

step = 20
numVideos = 50

dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)
featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, 10, step, 5, 4)
featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, 10, step, 5, 4)

featsTrain = preprocessing.scale(featsTrain)
featsTest = preprocessing.scale(featsTest)

baseSVM = svm.SVC(kernel='rbf', C=1000, class_weight='balanced')

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
            
