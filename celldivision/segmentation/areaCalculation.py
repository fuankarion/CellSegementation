import cv2
import math
import numpy as np

im = cv2.imread('./r152.jpg', 0)
"""
cv2.imshow('image',im)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

print(im.shape)
im[im > 1] = 1

#hist, bin_edge = np.histogram(im, bins=[x for x in range(0, 260)])
#print('hist', hist)

nonZero = np.count_nonzero(im)
print('nonZero ', nonZero)
radius = math.sqrt(nonZero / math.pi)
print('Approx radius', radius)
