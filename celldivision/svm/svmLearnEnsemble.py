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

###Optimal param
step = 10
voxelXYSize = 10
voxelTimeSize = 5
derivativeOrder = 4
kernelOpt = 'rbf'
COpt = 1000

timeStep = 10
tolerance = 0

dirsTrain = ['E37', 'E40', 'E82', 'E56', 'E81', 'E99', 'E92', 'E66', 'E88',
    'E08', 'E45', 'E10', 'E32', 'E77', 'E42', 'E01', 'E57', 'E62',
    'E65', 'E14', 'E85', 'E84', 'E27', 'E94', 'E63', 'E02', 'E00',
    'E26', 'E78', 'E11', 'E50', 'E25', 'E13', 'E80', 'E03', 'E86',
    'E68', 'E19', 'E90', 'E06', 'E15', 'E18', 'E33', 'E69', 'E51',
    'E21', 'E41', 'E16', 'E48', 'E74', 'E30', 'E73', 'E46', 'E09',
    'E89', 'E76', 'E61', 'E28', 'E29', 'E04', 'E70', 'E38', 'E98',
    'E12', 'E75', 'E91', 'E95', 'E55', 'E47', 'E83']
    
dirsTest = ['E59', 'E52', 'E53', 'E93', 'E17', 'E44', 'E60', 'E64', 'E72',
    'E20', 'E39', 'E96', 'E36', 'E24', 'E71', 'E22', 'E35', 'E43',
    'E31', 'E23', 'E97', 'E67', 'E79', 'E54', 'E05', 'E34', 'E07',
    'E49', 'E87', 'E58']
    

featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelXYSize, voxelTimeSize, step, timeStep, derivativeOrder, True, tolerance, True)
featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelXYSize, voxelTimeSize, step, timeStep, derivativeOrder, True, tolerance, True)

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
            
joblib.dump(clf, '/home/jcleon/Storage/ssd0/cellDivision/models/svmEnsembleOkSets.pkl') 
