import multiprocessing as mp
from random import shuffle
import re
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from utils import *

datasetRoot = '/home/jcleon/Storage/ssd1/cellDivision/MouEmbTrkDtb'
processPool = mp.Pool(16)

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def createTrainAndTestSubSets(datasetRoot, numVideos):
    dirsTrain = []
    dirsTest = []
    
    allDirs = []
    
    videoDirs = os.listdir(datasetRoot)
    shuffle(videoDirs)

    for aVideoDir in videoDirs:
        allDirs.append(aVideoDir)
        if len(allDirs)  > numVideos:
            break
    shuffle(allDirs)
    
    dirsTrain = allDirs[:int(numVideos * 0.7)]
    dirsTest = allDirs[int(numVideos * 0.7) + 1:]
            
    dirsTrain = sorted(dirsTrain, key=natural_key)
    dirsTest = sorted(dirsTest, key=natural_key)
    print('dirsTrain ', dirsTrain)
    print('dirsTest ', dirsTest)
    return dirsTrain, dirsTest
 
def loadSetFromVideos(videoDirs, datasetRoot, voxelSize, step, timeSize, order):
    featsSet = None
    labelsSet = None

    featCalculationArgs = []
    for aVideoDir in videoDirs:
        dirFrames = os.path.join(datasetRoot, aVideoDir)
        videoCube = loadVideoCube(dirFrames)
        
        featCalculationArgs.append((videoCube, voxelSize, step, timeSize, order, aVideoDir, datasetRoot))
    
    print('process feat data')
    data = processPool.map(getTrainDataFromVideo, featCalculationArgs)
    
    print('Unroll feat data')
    featsSet = None
    labelsSet = None
    for aCubeResult in data:
        if featsSet == None:
            featsSet = aCubeResult[0]
            labelsSet = aCubeResult[1]
        else:
            featsSet = np.concatenate((featsSet, aCubeResult[0]), axis=0)
            labelsSet = np.concatenate((labelsSet, aCubeResult[1]), axis=0)
    return featsSet, labelsSet 




##Optimal Config
"""
bestTimeSize = 3
descriptorOrder = 4
BestC=50
bestVoxelSize=15
"""

step = 20
numVideos = 50

dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)

#Load Feats

for timeSize in [1, 2, 3, 5, 10, 15]:
    for voxelSize in range(5, 35, 5):
        for descriptorOrder in range(1, 5):
            featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelSize, step, timeSize, descriptorOrder)
            featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelSize, step, timeSize, descriptorOrder)
            
            featsTrain = preprocessing.scale(featsTrain)
            featsTest = preprocessing.scale(featsTest)
          
            print('Train Softmax')
            print('timeSize', timeSize)
            print('voxelSize', voxelSize)
            print('descriptorOrder', descriptorOrder)
            print('featsTrain.shape ', featsTrain.shape)
            print('featsTest.shape ', featsTest.shape)

            clf = SGDClassifier(loss="log", random_state=17, n_iter=1000)

            start = time.time()
            clf.fit(featsTrain, labelsTrain) 
            end = time.time()

            print('Softmax Train Time ', end - start)

            preds = clf.predict(featsTest)
            target_names = ['Background', 'Cell', 'Boundary']
            classificationReport = classification_report(labelsTest, preds, target_names=target_names)
            f1Skore = f1_score(labelsTest, preds, average='weighted')
            print(classificationReport)
        
            targetDir = '/home/jcleon/Storage/ssd1/cellDivision/fullEvalSmax/'
            joblib.dump(clf, targetDir + 'TS' + str(timeSize) + '-DO' + str(descriptorOrder) + '-VS' + str(voxelSize) + 'SvmModel.pkl') 
            with open(targetDir + str(numVideos) + 'VideosTrain.txt', 'a') as myfile:
                myfile.write('\nNumVideos: ' + str(numVideos) + '\nTS: ' + str(timeSize) + '\nDO: ' + str(descriptorOrder) + '\nVS: ' + str(voxelSize) + '\n' + classificationReport) 

            with open(targetDir + str(numVideos) + 'report.txt', 'a') as myfile:
                myfile.write(str(voxelSize) + ',' + str(timeSize) + ',' + str(descriptorOrder) + ',' + str(f1Skore) + '\n')
            

