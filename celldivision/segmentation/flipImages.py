import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]	

def flipImage(img):
    flipped = img.T
    return flipped

def flipDir(dirPath):
    files = glob.glob(dirPath + '/*.jpg')
    files = sorted(files, key=natural_key)
    
    flippedImgs = []
    for aFile in files:
        img = cv2.imread(os.path.join(dirPath, aFile),0)
        flippedIm = flipImage(img)
        flippedImgs.append(flippedIm)
    
    return flippedImgs

def flipSet(videoSetPath, targetPath):
    dirs = os.listdir(videoSetPath)
    print('Videos ', len(dirs))
    for aDir in dirs:
        sourcePath = os.path.join(videoSetPath, aDir)
        flippedImgs = flipDir(sourcePath)
        
        targetDir = os.path.join(targetPath, aDir)
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
            print ('created ', targetDir)
        
        idx = 0
        
        for aFlippedImg in flippedImgs:
            cv2.imwrite(os.path.join(targetDir, str(idx) + '.jpg'), aFlippedImg)
            
flipSet('/home/jcleon/Storage/ssd0/cellDivision/segmentations', '/home/jcleon/Storage/ssd0/cellDivision/flippedSegmentations')