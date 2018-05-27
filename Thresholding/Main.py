import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import Binarization, Remove_Calibration, Segmentation



def showImage(image, title):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()


def pre_processing(image_file, file):

    biggest_component = Remove_Calibration.cutout_img(image_file)

    bin_image = Binarization.binarize(biggest_component)
    Binarization.save_image(bin_image, file.split(".")[0])              # Optional - Just to save the output

    cropped_charters = Segmentation.segmentation(bin_image)
    Segmentation.save_segmented_characters(cropped_charters, file)      # Optional - Just to save the output

    return cropped_charters

def main():

    mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        file_name = mypath + '\\' + file

        # Calling all the preprocessors from here
        # Here file is the name of the image file filename is the whole directory to that file
        cropped_charters = pre_processing(file_name, file)

        # Call to Classifier Using the cropped_characters (a list of segmented characters per image)



if __name__== "__main__":
  main()
