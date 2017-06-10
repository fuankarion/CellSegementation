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
numVideos = 6

###Optimal param
step = 20
voxelXYSize = 10
voxelTimeSize = 5
derivativeOrder = 4
kernelOpt = 'rbf'
COpt = 1000

timeStep = 4
tolerance = 5

dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)
featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelXYSize, voxelTimeSize, step, timeStep, derivativeOrder, True, tolerance, False)
featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelXYSize, voxelTimeSize, step, timeStep, derivativeOrder, True, tolerance, False)

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
target_names = ['Cell', 'Boundary']
classificationReport = classification_report(labelsTest, preds, target_names=target_names)
print(classificationReport)
            
joblib.dump(clf, '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleCheat.pkl') 
