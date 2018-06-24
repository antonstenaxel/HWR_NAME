import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import sys

if(len(sys.argv) > 1):
    mypathpath = sys.argv[1]
else:
    mypath = 'images/New folder'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for file in onlyfiles:
    file_name = mypath + os.sep + file
    originalImage = cv2.imread(file_name)


    img_copy = np.copy(originalImage)
    img_area = np.shape(originalImage)[0]*np.shape(originalImage)[1]
    treshold = 30
    originalImage[originalImage<=treshold] = 0
    #originalImage[originalImage>treshold] = 1
    binaryImage = originalImage.astype(np.uint8)


    blur = cv2.GaussianBlur(binaryImage, (5, 5), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im_contour = cv2.drawContours(blur, contours, -1, (0, 255, 0), 3)

    plt.imshow(im_contour, cmap=plt.cm.gray)
    #plt.title(title)
    plt.axis('off')
    plt.show()
