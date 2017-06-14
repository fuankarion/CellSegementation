from sklearn.externals import joblib
import os
import imutils
import cv2
import numpy as np
from scipy.cluster.vq import *
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


dataBOW = joblib.load("/home/jcleon/DAKode/CellSegmentation/celldivision/stageClassification/bof.pkl")
stdSlr=dataBOW[2]
k=dataBOW[3]
voc=dataBOW[4]
clf=dataBOW[0]
featureDetector = cv2.xfeatures2d.SIFT_create()

testPath = '/home/jcleon/Storage/disk0/Stages/Stages/testSmall'
testNames = os.listdir(testPath)

imagePaths = []
imageClasses = []
class_id = 0

print('Load images')
for training_name in testNames:
    dir = os.path.join(testPath, training_name)
    class_path = imutils.imlist(dir)
    imagePaths += class_path
    imageClasses += [class_id] * len(class_path)
    class_id += 1    
    
desList = []
print('Calculate descriptors')
for anImagePath in imagePaths:
    im = cv2.imread(anImagePath)
    kpts = featureDetector.detect(im)
    kpts, des = featureDetector.compute(im, kpts)
    desList.append(des)   

print('Stack descriptors')
descriptors = np.vstack(desList)

print('Histogram Construction ')
imFeatures = np.zeros((len(desList), k), "float32")
for i in xrange(len(desList)):
    words, distance = vq(desList[i], voc)
    for w in words:
        imFeatures[i][w] += 1

print('Prediction')
preds = clf.predict(imFeatures)
classificationReport = classification_report(np.array(imageClasses), preds)
print(classificationReport)
