import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import Binarization, Remove_Calibration, Segmentation
from baseline_cnn_classifier import Classifier
import sys


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

    # Returns a list of segmented characters and number of rows
    cropped_characters, row = Segmentation.segmentation(bin_image)
    Segmentation.save_segmented_characters(cropped_characters, file)      # Optional - Just to save the output

    return cropped_characters, row

def main():

    mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        file_name = mypath + os.sep + file

        # Calling all the preprocessors from here
        # Here file is the name of the image file filename is the whole directory to that file
        cropped_characters, row = pre_processing(file_name, file)
        #for character in cropped_characters:
            #print(character)


        # Call to Classifier Using the cropped_characters (a list of segmented characters per image)
        # print(type(cropped_characters[0]))

        cf = Classifier(path_to_model = "baseline_cnn.h5")

        # if(len(sys.argv) > 1):
            # image_path = sys.argv[1]
        for c, character in enumerate(cropped_characters):
            showImage(character, c)
            print(c)
            pred = cf.predict(img = character, print_result=True)
            



if __name__== "__main__":
  main()
