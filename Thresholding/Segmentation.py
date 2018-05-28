from skimage import filters, segmentation, io
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
import sys, os
from os import listdir
from os.path import isfile, join
from skimage.color import rgb2gray
from skimage.morphology import diamond,disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
import operator
import scipy
from scipy import ndimage as ndi
import math


def image_to_nparray(image):
    #print(type(image))
    data = np.asarray( image, dtype="int32" )
    return data


def nparray_to_image(npdata):
    img = scipy.misc.toimage(npdata)
    return img


def segmentation(image):

    # print(image_file)
    # im = rgb2gray(io.imread(image_file))
    # im = ndimage.imread(image_file)

    im = np.array(image).astype(float)

    plots_to_show = []

    ############################
    # X-Y axis pixel dilations #
    ############################

    # plot the amount of white ink across the columns & rows
    row_vals = list([sum(r) for r in im  ])
    col_vals = list([sum(c) for c in im.T])

    if "col_sums" in plots_to_show:
        plt.plot(col_vals)
        plt.show()

    if "row_sums" in plots_to_show:
        plt.plot(row_vals)
        plt.show()

    #########################################
    # Otsu/Sauvola method of boolean classification #
    #########################################

    val = filters.threshold_otsu(im)
    mask = im < val

    # val = filters.threshold_sauvola(im, window_size=21)
    # mask = im < val

    clean_border = clear_border(mask)

    # plt.imshow(clean_border, cmap='gray')
    # plt.show()

    # Opening Operation
    selem = disk(2)                                         #A disk of 2 pixel as structuring element
    clean_border = opening(clean_border, selem)


    #######################
    # Label image regions #
    #######################

    labeled = label(clean_border)
    image_label_overlay = label2rgb(labeled, image=im)

    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(image_label_overlay)
    #plt.show()

    #########################################
    # Draw bounding box around each article #
    #########################################

    # create array in which to store cropped articles
    cropped_images = []

    # define amount of padding to add to cropped image
    pad = 0

    # Sorting But failed...
    region_bbox = []

    for region_index, region in enumerate(regionprops(labeled)):
        if region.area < 200:
            continue

        # For sorting purpose by row and column from right to left
        minr, minc, maxr, maxc = region.bbox
        # (x,y) are the centroid
        x = math.ceil((minr+maxr)/2.0)
        y = math.ceil((minc+maxc)/2.0)
        # Number of rows
        row_no = int(x/100)

        region_bbox.append([row_no, x, y, minr, minc, maxr, maxc])
        # region_bbox.append(region.bbox)

    region_bbox = sorted(region_bbox, key = lambda x: (x[0], -x[2]))

    # region_bbox = sorted(region_bbox, key = lambda x: (x[3]))

    for region in region_bbox:
        #print(region)

        row_no, x, y, minr, minc, maxr, maxc = region

        # use those bounding box coordinates to crop the image
        cropped = nparray_to_image(im[minr-pad:maxr+pad, minc-pad:maxc+pad])
        cropped_images.append(cropped)

        # print ("region", region_index, "bounding box:", minr, minc, maxr, maxc)

        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)

        ax.add_patch(rect)

    # plt.show()

    return cropped_images


def save_segmented_characters(cropped_images, file):

    ###############
    # Crop images #
    ###############

    out_dir = "segmented_articles/"+file+"/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # can crop using: cropped = image_array[x1:x2,y1:y2]

    for c, cropped_image in enumerate(cropped_images):
        io.imsave( out_dir + str(c) + ".jpg", cropped_image)


# def main():
#
#     mypath = 'segmented'
#     onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#
#     for file in onlyfiles:
#         cropped_images = segmentation(mypath+'\\'+file)
#         save_segmented_characters(cropped_images, file.split(".")[0])
#
#
# if __name__== "__main__":
#   main()
