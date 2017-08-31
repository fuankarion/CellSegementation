import cv2
import imutils
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


videosTest = ['E59', 'E52', 'E53', 'E93', 'E17', 'E44', 'E60', 'E64', 'E72',
    'E20', 'E39', 'E96', 'E36', 'E24', 'E71', 'E22', 'E35', 'E43',
    'E31', 'E23', 'E97', 'E67', 'E79', 'E54', 'E05', 'E34', 'E07',
    'E49', 'E87', 'E58']
    
videosT = ['E59']
    
def exploreParameterSet(k, descriptors, descriptorList, targetPath):
    # Perform k-means clustering
    print('Clustering k=', k)
    voc, variance = kmeans(descriptors, k, 1) 

    # Calculate the histogram of features
    print('Histogram Construction ')
    imFeatures = np.zeros((len(descriptorList), k), "float32")
    for i in xrange(len(descriptorList)):
        words, distance = vq(descriptorList[i], voc)
        for w in words:
            imFeatures[i][w] += 1

    # Perform Tf-Idf vectorization
    nbr_occurences = np.sum((imFeatures > 0) * 1, axis=0)
    idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

    # Scaling the words
    stdSlr = StandardScaler().fit(imFeatures)
    imFeatures = stdSlr.transform(imFeatures)

    # Train the Linear SVM
    print('Train SVM')
    clf = SVC(class_weight='balanced')
    clf.fit(imFeatures, np.array(image_classes))
    preds = clf.predict(imFeatures)
    classificationReport = classification_report(np.array(image_classes), preds)
    print(classificationReport)

    # Save the SVM
    joblib.dump((clf, trainingNames, stdSlr, k, voc), os.path.join(targetPath, 'stageBOW' + str(k) + '.pkl'))    


featureDetector = cv2.xfeatures2d.SIFT_create()

# Get the training classes names and store them in a list
targetPath = '/home/jcleon/Storage/ssd0/cellDivision/models'
trainPath = '/home/jcleon/Storage/disk2/cellDivision/MouEmbTrkDtb'
trainingNames = os.listdir(trainPath)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

print('Load images')
for training_name in trainingNames:
    if training_name in videosTest:
        dir = os.path.join(trainPath, training_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
        image_classes += [class_id] * len(class_path)
        class_id += 1
print('A total of ', len(image_paths), ' images')

# List where all the descriptors are stored
descriptorList = []
print('Calculate descriptors')
for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts = featureDetector.detect(im)
    kpts, des = featureDetector.compute(im, kpts)
    descriptorList.append(des)      
    
# Stack all the descriptors vertically in a numpy array
print('Stack descriptors')
descriptors = np.vstack(descriptorList)


for k in range(120, 140, 10):
    exploreParameterSet(k, descriptors, descriptorList, targetPath)