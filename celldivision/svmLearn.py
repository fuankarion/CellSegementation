import multiprocessing as mp
from random import shuffle
import re
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
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

voxelSize = 10
step = 20
#timeSize = 1
order = 4
CValue = 100

for numVideos in range(3, 100, 3):
    for timeSize in range(1,10,2):
        print('numVideos', numVideos)
        dirsTrain, dirsTest = createTrainAndTestSubSets(datasetRoot, numVideos)

        #Load Feats
        featsTrain, labelsTrain = loadSetFromVideos(dirsTrain, datasetRoot, voxelSize, step, timeSize, order)
        featsTest, labelsTest = loadSetFromVideos(dirsTest, datasetRoot, voxelSize, step, timeSize, order)

        print('featsTrain.shape ', featsTrain.shape)
        print('featsTest.shape ', featsTest.shape)

        print('Train SVM')
        baseSVM = svm.SVC(kernel='rbf', C=CValue, class_weight='balanced')

        numEstimators = 10
        clf = BaggingClassifier(baseSVM, n_estimators=numEstimators, max_samples=1.0 / numEstimators, n_jobs=numEstimators)
        start = time.time()
        clf.fit(featsTrain, labelsTrain) 
        end = time.time()
        print('SVM Train Time ', end - start)

        preds = clf.predict(featsTest)
        target_names = ['Background', 'Cell', 'Boundary']
        classificationReport = classification_report(labelsTest, preds, target_names=target_names)
        print(classificationReport)

        targetDirTS='/home/jcleon/Storage/ssd1/cellDivision/classifiers'+str(timeSize)+'TS/'
        if not os.path.exists(targetDirTS):
            os.makedirs(targetDirTS)
            print ('created ', targetDirTS)
        
        joblib.dump(clf, targetDirTS + str(numVideos) + 'VideosTrain.pkl') 
        with open(targetDirTS + str(numVideos) + 'VideosTrain.txt', 'a') as myfile:
            myfile.write('NumVideos: ' + str(numVideos) + '\n' + 'C: ' + str(CValue) + '\n' + classificationReport) 

