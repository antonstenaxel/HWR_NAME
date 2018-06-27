import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import sys
import scipy
from collections import Counter

def nparray_to_image(npdata):
    img = scipy.misc.toimage(npdata)
    return img

if(len(sys.argv) > 1):
    mypathpath = sys.argv[1]
else:
    mypath = 'images/all_images'
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

    #Blur the image to reduce the noise
    blur = cv2.GaussianBlur(binaryImage, (5, 5), 0)

    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    # print(contours)
    biggestContour = max(contours, key = cv2.contourArea)
    BCLength= len(biggestContour)
    # print(len(biggestContour), len(biggestContour[0]), len(biggestContour[0][0]))
    print("biggest Contour", biggestContour)
    smallestContour = min(contours, key = cv2.contourArea)
    SCLength = len(smallestContour)
    print("smallest Contour ", smallestContour)

    mask = np.ones(originalImage.shape[:2], dtype="uint8") * 255
    # mask = np.zeros_like(originalImage)
    im_contour = cv2.drawContours(blur, biggestContour, -1, (0, 255, 0), 3)
    out = np.zeros_like(im_contour) # Extract out the object and place into output image
    out[im_contour == 255] = originalImage[im_contour == 255]

    # loop over the contours
    for c in contours:

    # if the contour is bad, draw it on the mask
        if SCLength < BCLength:
            contours.remove(smallestContour)
            cv2.drawContours(mask, smallestContour, -1, 0, 3)

    # remove the contours from the image and show the resulting images
            image = cv2.bitwise_and(blur, blur, mask=mask)

            plt.imshow(image, cmap=plt.cm.gray)
            #plt.title(title)
            plt.axis('off')
            plt.show()
