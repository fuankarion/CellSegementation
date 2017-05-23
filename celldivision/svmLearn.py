from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from utils import *

datasetRoot = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb'

voxelSize = 10
step = 15
timeSize = 1
order = 4

dirsTrain = ['E01', 'E02']
dirsTest = ['E03', 'E04']



def loadSetFromVideos(videoDirs):
    featsSet = None
    labelsSet = None

    for aVideoDir in videoDirs:
        dirFrames = os.path.join(datasetRoot, aVideoDir)
        videoCube = loadVideoCube(dirFrames)

        print('Extract Feats ', aVideoDir)
        feats, labels = getTrainDataFromVideo(videoCube, voxelSize, step, timeSize, order, aVideoDir)

        if featsTrain == None:
            featsSet = feats
            labelsSet = labels
        else:
            featsSet = np.concatenate((featsSet, feats), axis=0)
            labelsSet = np.concatenate((labelsSet, labels), axis=0)
    return featsSet, labelsSet

featsTrain, labelsTrain = loadSetFromVideos(dirsTrain)
featsTest, labelsTest = loadSetFromVideos(dirsTest)

print('Train SVM')
baseSVM = svm.SVC(kernel='rbf', C=1, class_weight='balanced')
baseSVM.fit(featsTrain, labelsTrain) 

preds = baseSVM.predict(featsTest)
target_names = ['Background', 'Cell', 'Boundary']
print(classification_report(labelsTest, preds, target_names=target_names))
