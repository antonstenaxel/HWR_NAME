import numpy as np
import os, cv2, sys
import matplotlib.pyplot as plt

def binarize(_img,treshold = 30):
    _img[_img<=treshold] = 0
    _img[_img>treshold] = 1
    return _img.astype(np.uint8)

def cutout_img(image_file):

    filename = os.path.join(image_file)
    _img = cv2.imread(filename,0)

    img_copy = np.copy(_img)
    img_area = np.shape(_img)[0]*np.shape(_img)[1]
    binarized_image = binarize(_img)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)
    centroids = output[3]
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    lista = np.argsort(stats[:,cv2.CC_STAT_AREA])

    # Go through components sorted by area and check if bbox contains img center
    for i in lista[::-1]:
        left,top,width,height = stats[i,0:4]

        x1 = left
        x2 = left + width
        y1 = top
        y2 = top + height

        x,y = centroids[i]
        # print(stats[i,cv2.CC_STAT_AREA],img_area)
        if( x > x1 and x < x2 and y > y1 and y < y2 and (x1 != 0 or y1 != 0)):
            right_component = i
            break

    img_copy[labels != right_component] = 0
    # Bounding box of biggest component
    left,top,width,height = stats[right_component,0:4]
    new_img = img_copy[top:top+height,left:left+width]


    return new_img

