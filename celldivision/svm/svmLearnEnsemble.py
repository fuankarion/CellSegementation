from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import classification_report
import sys
sys.path.append('../')
from utils import *
from svmUtil import *
from sklearn.externals import joblib

datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
numVideos = 100

###Optimal param
step = 20
voxelXY = 10
timeRange = 5
derivativeOrder = 4
kernelOpt = 'rbf'
COpt = 1000

tolerance = 0

dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)
featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelXY, step, timeRange, derivativeOrder, True, tolerance, True)
featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelXY, step, timeRange, derivativeOrder, True, tolerance, True)

featsTrain = preprocessing.scale(featsTrain)
featsTest = preprocessing.scale(featsTest)

baseSVM = svm.SVC(kernel=kernelOpt, C=COpt, class_weight='balanced')
numEstimators = 10
clf = BaggingClassifier(baseSVM, n_estimators=numEstimators, max_samples=1.0 / numEstimators, n_jobs=numEstimators)

print('Start SVM Train')
start = time.time()
clf.fit(featsTrain, labelsTrain) 
end = time.time()
print('SVM Train Time ', end - start)

preds = clf.predict(featsTest)
target_names = ['Background', 'Cell', 'Boundary']
classificationReport = classification_report(labelsTest, preds, target_names=target_names)
print(classificationReport)
            
joblib.dump(clf, '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleCheat.pkl') 
