import os
from os import listdir
from os.path import isfile, join
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from skimage.data import page
from skimage.color import rgb2gray
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

def thresholding(image_name):

    # matplotlib.rcParams['font.size'] = 9
    # image = page()
    filename = os.path.join(image_name)
    image = io.imread(filename)
    # image = img_as_ubyte(image)
    # image = rgb2gray(io.imread(filename))


    binary_global = image > threshold_otsu(image)

    # window_size = 21
    thresh_niblack = threshold_niblack(image, window_size=101, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=21)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola



    showImage(image, 'Original')
    showImage(binary_global, 'Global Threshold (OTSU)')
    # showImage(binary_niblack, 'Niblack Threshold')
    # showImage(binary_sauvola, 'Sauvola Threshold')


    #Applying morphological operation
    selem = disk(2)                                         #A disk of 2 pixel as structuring element
    closed_niblack = closing(binary_niblack, selem)
    closed_sauvola = closing(binary_sauvola, selem)
    selem = disk(2)
    opened_niblack = opening(closed_niblack, selem)
    opened_sauvola = opening(closed_sauvola, selem)


    showImage(closed_niblack, 'Niblack (Closing)')
    showImage(opened_niblack, 'Niblack (Opening)')
    showImage(closed_sauvola, 'Sauvola (Closing)')
    showImage(opened_sauvola, 'Sauvola (Opening)')


def showImage(image, title):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def main():

    mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for files in onlyfiles:
        thresholding(mypath+'\\'+files)

    # thresholding('sample1.jpg')

if __name__== "__main__":
  main()
