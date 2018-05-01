# This is only to test... Input an image and how to output it.

import os
import skimage
from skimage import io
import matplotlib
import matplotlib.pyplot as plt
from skimage.viewer import ImageViewer
from skimage import data


def main():
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join('sample1.jpg')
    image = io.imread(filename)

    # image = data.coins()
    print(filename)
    # plt.imshow(image)

    # image = data.coins()
    # viewer = ImageViewer(image)
    # viewer.show()

    plt.figure(figsize=(8, 7))
    # plt.subplot(2, 2, 2)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title('Original')
    plt.axis('off')
    plt.show()

if __name__== "__main__":
  main()
