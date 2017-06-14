import cv2
import imutils
import numpy as np
import os
from scipy.cluster.vq import *
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def exploreParameterSet(k, descriptors, descriptorList):
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
    joblib.dump((clf, trainingNames, stdSlr, k, voc), 'bof'+str(k)+'.pkl', compress=3)    


featureDetector = cv2.xfeatures2d.SIFT_create()

# Get the training classes names and store them in a list
trainPath = '/home/jcleon/Storage/disk0/Stages/Stages/trainSmall'
trainingNames = os.listdir(trainPath)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

print('Load images')
for training_name in trainingNames:
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


for k in range(10,200,10):
    exploreParameterSet(k, descriptors, descriptorList)