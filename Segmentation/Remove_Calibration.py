import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


def cutout_img(image_file):

    file_name = os.path.join(image_file)

    originalImage = cv2.imread(file_name,0)

    img_copy = np.copy(originalImage)

    img_copy = img_copy.astype(np.uint8)

    #Blur the image to reduce the noise
    blur = cv2.GaussianBlur(img_copy, (5, 5), 0)

    # imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(blur, 30, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area = 0
    bounding_rect = cv2.boundingRect(contours[0])

    for i in contours:
        area = cv2.contourArea(i)
        if (area>largest_area):
            largest_area=area
            largest_contour_index=i
            bounding_rect=cv2.boundingRect(i)

    # Crop image
    r = bounding_rect
    imCrop = originalImage[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # plt.imshow(imCrop, cmap=plt.cm.gray)
    # plt.show()

    return imCrop


def save_image(image, file_name):

    out_dir = "biggest/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.imsave(out_dir + file_name + ".jpg", image, cmap=plt.cm.gray)
    # io.imsave( out_dir + file_name + ".pbm", image)
