import os, random, cv2
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
from skimage.filters import threshold_otsu
from skimage import exposure

def binarize(image):

    # filename = os.path.join(image_file)
    # image = rgb2gray(io.imread(filename))

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

    # image = image < threshold_otsu(image)
    # showImage(image, "")

    # Change the contrast of the Image
    # v_min, v_max = np.percentile(image, (1.5, 100.0))
    # image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    # showImage(image, "")

    # Produce a threshold to make the marker
    markers = np.zeros_like(image)
    threshold = threshold_otsu(image)
    markers[image < threshold] = 1
    markers[image > threshold] = 2

    # markers[image < 30] = 1
    # markers[image > 80] = 2

    elevation_map = sobel(image)
    segmentation = watershed(elevation_map, markers)

    return segmentation
    # return segmentation.astype(np.uint8)


def save_image(image, file_name):

    out_dir = "binarized/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.imsave(out_dir + file_name + ".jpg", image, cmap=plt.cm.gray)
    # io.imsave( out_dir + file_name + ".pbm", image)


# def main():
#
#     mypath = 'images'
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#     for file in onlyfiles:
#         seg_image = binarize(mypath+'\\'+file)
#         save_image(seg_image, file.split(".")[0])
#
# if __name__== "__main__":
#   main()
