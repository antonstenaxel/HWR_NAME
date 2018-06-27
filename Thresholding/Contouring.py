import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import sys
import scipy
from skimage import io
import Segmentation, Binarization



def nparray_to_image(npdata):
    img = scipy.misc.toimage(npdata)
    return img

def is_contour_bad(c):
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	# the contour is 'bad' if it is not a rectangle
	return not len(approx) == 4

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
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE)
    # print("Contours", contours)
    cnt = contours[0]
    biggestContour = max(contours, key = cv2.contourArea)
    print("biggest Contour", biggestContour)
    smallestContour = min(contours, key = cv2.contourArea)
    print("smallest Contour", smallestContour)

    mask = np.ones(originalImage.shape[:2], dtype="uint8") * 255
    # mask = np.zeros_like(originalImage)
    im_contour = cv2.drawContours(blur, biggestContour, -1, (0, 255, 0), 3)
    out = np.zeros_like(im_contour) # Extract out the object and place into output image
    out[im_contour == 255] = originalImage[im_contour == 255]

# loop over the contours
    for c in contours:
        # print("each contour", c[0])

        # print ("smallest contour", smallestContour)
    # if the contour is bad, draw it on the mask

        if smallestContour == biggestContour:
            cv2.drawContours(mask, biggestContour, -1, 0, 3)

    # remove the contours from the image and show the resulting images
            image = cv2.bitwise_and(blur, blur, mask=mask)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("After", image)
    # cv2.waitKey(0)

            plt.imshow(image, cmap=plt.cm.gray)
            #plt.title(title)
            plt.axis('off')
            plt.show()


    # # Now crop
    # (x, y) = np.where(im_contour == 255)
    # (topx, topy) = (np.min(x), np.min(y))
    # (bottomx, bottomy) = (np.max(x), np.max(y))
    # out = out[topx:bottomx+1, topy:bottomy+1]
    #
    # Show the output image
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()







