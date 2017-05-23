from utils import *
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

dirFramesTrain = os.path.join(datasetRoot, 'E01')
dirFramesTest = os.path.join(datasetRoot, 'E17')

videoCubeTrain = loadVideoCube(dirFramesTrain)  
videoCubeTest = loadVideoCube(dirFramesTest)  

voxelSize = 10
step = 15
timeSize = 1
order = 4

print('Extract Feats')
featsTrain, labelsTrain = getTrainDataFromVideo(videoCubeTrain, voxelSize, step, timeSize, order, 'E01')
featsTest, labelsTest = getTrainDataFromVideo(videoCubeTest, voxelSize, step, timeSize, order, 'E17')

print('Train SVM')
baseSVM = svm.SVC(kernel='rbf', C=1, class_weight='balanced')
baseSVM.fit(featsTrain, labelsTrain) 

y_pred = baseSVM.predict(featsTest)
target_names = ['Background', 'Cell', 'Boundary']
print(classification_report(labelsTest, y_pred, target_names=target_names))
