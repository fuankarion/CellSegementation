from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
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

for timeSize in [1, 2, 3, 5, 10, 15]:
    for voxelSize in range(5, 35, 5):
        for descriptorOrder in range(1, 5):
            featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelSize, step, timeSize, descriptorOrder)
            featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelSize, step, timeSize, descriptorOrder)
            
            featsTrain = preprocessing.scale(featsTrain)
            featsTest = preprocessing.scale(featsTest)
            
            for CValue in [0.1, 1, 10, 100, 1000]:
                for kernelT in ['rbf', 'linear', 'poly']:
                    print('Train SVM')
                    print('timeSize', timeSize)
                    print('voxelSize', voxelSize)
                    print('descriptorOrder', descriptorOrder)
                    print('CValue', CValue)
                    print('kernelT', kernelT)
                    print('featsTrain.shape ', featsTrain.shape)
                    print('featsTest.shape ', featsTest.shape)
                    
                    if kernelT == 'rbf':
                        baseSVM = svm.SVC(kernel=kernelT, C=CValue, class_weight='balanced')
                    else:
                        baseSVM = svm.SVC(kernel=kernelT, C=CValue, class_weight='balanced', max_iter=200000)

                    numEstimators = 10
                    clf = BaggingClassifier(baseSVM, n_estimators=numEstimators, max_samples=1.0 / numEstimators, n_jobs=numEstimators)
                    start = time.time()
                    clf.fit(featsTrain, labelsTrain) 
                    end = time.time()
                    
                    print('SVM Train Time ', end - start)

                    preds = clf.predict(featsTest)
                    target_names = ['Background', 'Cell', 'Boundary']
                    classificationReport = classification_report(labelsTest, preds, target_names=target_names)
                    f1Skore = f1_score(labelsTest, preds, average='weighted')
                    print(classificationReport)

                    targetDir = '/home/jcleon/Storage/ssd1/cellDivision/fullEval/'
                    #joblib.dump(clf, targetDir + 'TS' + str(timeSize) + '-DO' + str(descriptorOrder) + '-CV' + str(CValue) + '-VS' + str(voxelSize) + '-KT' + kernelT + 'SvmModel.pkl') 
                    with open(targetDir + str(numVideos) + 'VideosTrain.txt', 'a') as myfile:
                        myfile.write('\nNumVideos: ' + str(numVideos) + '\nTS: ' + str(timeSize) + '\nDO: ' + str(descriptorOrder) + '\nCV: ' + str(CValue) + '\nKT: ' + str(kernelT) + '\nVS: ' + str(voxelSize) + '\n' + classificationReport) 
                        
                    with open(targetDir + str(numVideos) + 'report.txt', 'a') as myfile:
                        myfile.write(str(voxelSize) + ',' + str(timeSize) + ',' + str(descriptorOrder) + ',' + str(CValue) + ',' + str(kernelT) + ',' + str(f1Skore) + '\n') 
