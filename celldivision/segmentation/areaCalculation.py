import cv2
import math
import numpy as np

im = cv2.imread('/home/jcleon/DAKode/CellSegmentation/celldivision/segmentation/r216.jpg', 0)

print(im.shape)
im[im > 1] = 0 
nonZero = np.count_nonzero(im)
print('nonZero ', nonZero)
radius = math.sqrt(nonZero / math.pi)
print('radius', radius)
