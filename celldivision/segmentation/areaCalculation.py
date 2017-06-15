import cv2
import math
import numpy as np

im = cv2.imread('/home/jcleon/Storage/ssd0/cellDivision/segmentations/E61/1.jpg', 0)

print(im.shape)
im[im < 250] = 0 
nonZero = np.count_nonzero(im)
print('nonZero ', nonZero)
radius = math.sqrt(nonZero / math.pi)
print('radius', radius)