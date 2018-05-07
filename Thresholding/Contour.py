import os
from os import listdir
from os.path import isfile, join
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
import cv2
import numpy as np

import scipy
from scipy import ndimage
from skimage.measure import label


def border_removal(image_name):

    filename = os.path.join(image_name)
    image = io.imread(filename)
    # image = rgb2gray(io.imread(filename))

    # apply threshold
    # thresh = threshold_otsu(image)
    # thresh = threshold_sauvola(image, window_size=21)
    # bw = closing(image > thresh, square(3))

    # remove artifacts connected to image border
    # cleared = clear_border(bw)
    cleared = clear_border(image)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(label_image)

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def bcc(image_name):

    filename = os.path.join(image_name)
    image = rgb2gray(io.imread(filename))

    blur_radius = 1.0
    threshold = 100

    # smooth the image (to remove small objects)
    imgf = ndimage.gaussian_filter(image, blur_radius)

    # find connected components
    labeled, nr_objects = ndimage.label(imgf > threshold)
    print("Number of objects is %d " % nr_objects)

    # plt.imsave('/tmp/out.png', labeled)
    plt.imshow(labeled)

    plt.show()

def getLargestCC(image_name):

    filename = os.path.join(image_name)
    image = io.imread(filename)

    labels = label(image)
    largestCC = labels == np.argmax(np.bincount(labels.flat))

    showImage(largestCC, "")

def undesired_objects (image_name):

    filename = os.path.join(image_name)
    # image = io.imread(filename)
    image = rgb2gray(io.imread(filename))
    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2[output == max_label] = 255
    cv2.imshow("Biggest component", img2)
    cv2.waitKey()

def showImage(image, title):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():

    mypath = 'binarized'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for files in onlyfiles:
        undesired_objects(mypath+'\\'+files)

    # thresholding('sample1.jpg')

if __name__== "__main__":
  main()
