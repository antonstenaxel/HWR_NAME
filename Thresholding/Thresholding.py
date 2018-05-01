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


def thresholding(image_name):

    # matplotlib.rcParams['font.size'] = 9

    # image = page()
    filename = os.path.join(image_name)
    image = rgb2gray(io.imread(filename))
    # image = img_as_ubyte(image)

    binary_global = image > threshold_otsu(image)

    # window_size = 21
    thresh_niblack = threshold_niblack(image, window_size=101, k=0.8)
    thresh_sauvola = threshold_sauvola(image, window_size=21)

    binary_niblack = image > thresh_niblack
    binary_sauvola = image > thresh_sauvola

    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')
    plt.show()

    # plt.subplot(2, 2, 2)
    plt.title('Global Threshold (OTSU)')
    plt.imshow(binary_global, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    # plt.subplot(2, 2, 3)
    plt.imshow(binary_niblack, cmap=plt.cm.gray)
    plt.title('Niblack Threshold')
    plt.axis('off')
    plt.show()

    # plt.subplot(2, 2, 4)
    plt.imshow(binary_sauvola, cmap=plt.cm.gray)
    plt.title('Sauvola Threshold')
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
