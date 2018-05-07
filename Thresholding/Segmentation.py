import os
from os import listdir
from os.path import isfile, join
from skimage import io
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
from skimage import data
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import img_as_uint
from skimage.filters import sobel
from skimage.morphology import watershed
from skimage.measure import label
from skimage.measure import regionprops

def segment(image_file):

    filename = os.path.join(image_file)
    image = rgb2gray(io.imread(filename))
    # image = io.imread(filename)

    # histo = np.histogram(image, bins=np.arange(0, 256))

    # edges = canny(image/255.)
    #
    # fill_coins = ndi.binary_fill_holes(edges)
    #
    # label_objects, nb_labels = ndi.label(fill_coins)
    # sizes = np.bincount(label_objects.ravel())
    # mask_sizes = sizes > 20
    # mask_sizes[0] = 0
    # coins_cleaned = mask_sizes[label_objects]

    markers = np.zeros_like(image)
    markers[image < 30] = 1
    markers[image > 150] = 2

    elevation_map = sobel(image)

    segmentation = watershed(elevation_map, markers)

    # showImage(segmentation, "")

    return segmentation


def showImage(image, title):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()


def save_image(image, file_name):

    out_dir = "segmented/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.imsave(out_dir + file_name + ".png", image, cmap=plt.cm.gray)
    # io.imsave( out_dir + file_name + ".png", image)

def main():

    mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for files in onlyfiles:
        seg_image = segment(mypath+'\\'+files)
        save_image(seg_image, files.split(".")[0])


if __name__== "__main__":
  main()
