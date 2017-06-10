import argparse as ap
import cv2
import imutils 
import numpy as np
import os
from sklearn.svm import LinearSVC,SVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

fea_det = cv2.xfeatures2d.SIFT_create()

# Get the path of the training set
parser = ap.ArgumentParser()
parser.add_argument("-t", "--trainingSet", help="Path to Training Set", required="True")
args = vars(parser.parse_args())

# Get the training classes names and store them in a list
train_path = args["trainingSet"]
training_names = os.listdir(train_path)

# Get all the path to the images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

print('Load images')
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imutils.imlist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# List where all the descriptors are stored
des_list = []
print('Calculate descriptors')
for image_path in image_paths:
#    print('image_path ',image_path)
    im = cv2.imread(image_path)
    kpts = fea_det.detect(im)
    kpts, des = fea_det.compute(im, kpts)
    des_list.append(des)      
    
# Stack all the descriptors vertically in a numpy array
print('Stack descriptors')
"""
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  
"""
descriptors = np.vstack(des_list)


# Perform k-means clustering
k = 100
print('Kmeans ')
voc, variance = kmeans(descriptors, k, 1) 

# Calculate the histogram of features
im_features = np.zeros((len(des_list), k), "float32")
print('Histogram Construction ')
for i in xrange(len(des_list)):
    words, distance = vq(des_list[i],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

# Train the Linear SVM
clf = SVC()
clf.fit(im_features, np.array(image_classes))
preds = clf.predict(im_features)
classificationReport = classification_report(np.array(image_classes), preds)
print(classificationReport)

# Save the SVM
joblib.dump((clf, training_names, stdSlr, k, voc), "bof.pkl", compress=3)    
    
