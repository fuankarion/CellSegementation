import cv2
import glob
import imutils
import numpy as np
import os
import re
from scipy.cluster.vq import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#Best k=130 @ 0.77

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	


datasetRoot = '/home/jcleon/Storage/ssd0/cellDivision/MouEmbTrkDtb'
videosTest = ['E59', 'E52', 'E53', 'E93', 'E17', 'E44', 'E60', 'E64', 'E72',
    'E20', 'E39', 'E96', 'E36', 'E24', 'E71', 'E22', 'E35', 'E43',
    'E31', 'E23', 'E97', 'E67', 'E79', 'E54', 'E05', 'E34', 'E07',
    'E49', 'E87', 'E58']

def calculateDescriptors(im):
    kpts = featureDetector.detect(im)
    kpts, des = featureDetector.compute(im, kpts)
    
    return des

imagePaths = []
imageClasses = []

dataBOW = joblib.load("/home/jcleon/Storage/ssd0/cellDivision/models/stageBOW" + str(130) + ".pkl")
stdSlr = dataBOW[2]
k = dataBOW[3]
voc = dataBOW[4]
clf = dataBOW[0]

featureDetector = cv2.xfeatures2d.SIFT_create()
targetFile='/home/jcleon/Storage/ssd0/cellDivision/Stages/Results.csv'

for video in videosTest:
    
    
    videoDir = os.path.join(datasetRoot, video)
    pathGT = os.path.join(videoDir, '_trajectories.txt')
    
    content = None
    with open(pathGT) as f:
        content = f.readlines()
    
    frames = glob.glob(videoDir + '/*.png')
    frames = sorted(frames, key=natural_key)
    
    for frameIdx in  range(0, len(content)):
        frame=frames[frameIdx]
        print('frame ',frame)
        img = cv2.imread(frame)
        desc = calculateDescriptors(img)

        print('Histogram Construction ')
        imFeature = np.zeros((1, k), "float32")
        words, distance = vq(desc, voc)
        for w in words:
            imFeature[0][w] += 1
        pred = clf.predict(imFeature)


        gtLine = content[frameIdx]
        print(gtLine)
        tokens = gtLine.split('\t')

        gtStage = 0
        if int(tokens[2]) != 0 and int(tokens[5]) == 0 and int(tokens[8]) == 0:
            gtStage = 1

        if int(tokens[2]) == 0 and int(tokens[5]) != 0 and int(tokens[8]) != 0:
            gtStage = 2

        if int(tokens[5]) == 0 and int(tokens[8]) == 0 and int(tokens[11]) != 0 and int(tokens[14]) != 0 and int(tokens[17]) != 0 and int(tokens[20]) != 0:
            gtStage = 3


        print('pred', pred, ' gtStage ', gtStage)
        with open(targetFile, 'a') as myfile:
            myfile.write(video+','+str(frameIdx)+'.png,'+str(gtStage)+','+str(int(pred))+'\n')
