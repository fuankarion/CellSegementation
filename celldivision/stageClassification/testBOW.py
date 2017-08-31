import cv2
import imutils
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#Best k=130 @ 0.72

testPath = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb'
testNames = os.listdir(testPath)
videosTest = ['E59', 'E52', 'E53', 'E93', 'E17', 'E44', 'E60', 'E64', 'E72',
    'E20', 'E39', 'E96', 'E36', 'E24', 'E71', 'E22', 'E35', 'E43',
    'E31', 'E23', 'E97', 'E67', 'E79', 'E54', 'E05', 'E34', 'E07',
    'E49', 'E87', 'E58']

imagePaths = []
imageClasses = []
class_id = 0

print('Load images')
for training_name in testNames:
    if training_name in videosTest:
        dir = os.path.join(testPath, training_name)
        class_path = imutils.imlist(dir)
        imagePaths += class_path
        imageClasses += [class_id] * len(class_path)
        class_id += 1    
    
featureDetector = cv2.xfeatures2d.SIFT_create()
desList = []
print('Calculate descriptors')
for anImagePath in imagePaths:
    im = cv2.imread(anImagePath)
    kpts = featureDetector.detect(im)
    kpts, des = featureDetector.compute(im, kpts)
    desList.append(des)   
    

for k in range(10, 200, 10):
    dataBOW = joblib.load("/home/jcleon/Storage/ssd0/cellDivision/models/stageBOW" + str(k) + ".pkl")
    stdSlr = dataBOW[2]
    k = dataBOW[3]
    voc = dataBOW[4]
    clf = dataBOW[0]

    print('Histogram Construction ')
    imFeatures = np.zeros((len(desList), k), "float32")
    for i in xrange(len(desList)):
        words, distance = vq(desList[i], voc)
        for w in words:
            imFeatures[i][w] += 1

    print('Prediction k=', k)
    preds = clf.predict(imFeatures)
    classificationReport = classification_report(np.array(imageClasses), preds)
    print(classificationReport)
