import os
import sys
sys.path.append('Classification')
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from Segmentation import Binarization, Remove_Calibration, Segmentation
#from Classification.improved_classifier import Classifier
from Classification.improved_classifier import Classifier
import json
import numpy as np
dic = {0: 'Alef',
1: 'Ayin',
2: 'Bet',
3: 'Dalet',
4: 'Gimel',
5: 'He',
6: 'Het',
7: 'Kaf',
8: 'Kaf-final',
9: 'Lamed',
10: 'Mem',
11: 'Mem-medial',
12: 'Nun-final',
13: 'Nun-medial',
14: 'Pe',
15: 'Pe-final',
16: 'Qof',
17: 'Resh',
18: 'Samekh',
19: 'Shin',
20: 'Taw',
21: 'Tet',
22: 'Tsadi-final',
23: 'Tsadi-medial',
24: 'Waw',
25: 'Yod',
26: 'Zayin',
27: 'Multi-letter',
28: 'Noise'}


def showImage(image, title):
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()


def pre_processing(image_file, file):

    biggest_component = Remove_Calibration.cutout_img(image_file)
    #Remove_Calibration.save_image(biggest_component, file.split(".")[0])

    bin_image = Binarization.binarize(biggest_component)
    #Binarization.save_image(bin_image, file.split(".")[0])              # Optional - Just to save the output

    # Returns a list of segmented characters and number of rows
    cropped_characters, row = Segmentation.segmentation(bin_image)
    Segmentation.save_segmented_characters(cropped_characters, row, file)      # Optional - Just to save the output

    return cropped_characters, row

def main():
    #cf = Classifier(path_to_model = "Classification/Models/baseline_cnn.h5")
    cf = Classifier(path_to_model = "Classification/Models/thinned_and_augmented_cnn_v2.h5")

    if(len(sys.argv) > 1):
        mypathpath = sys.argv[1]
    else:
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

        # if(len(sys.argv) > 1):
            # image_path = sys.argv[1]
        predictions = []
        for c, (char, r) in enumerate(zip(cropped_characters[:5], row[:5])):
            #plt.imshow(char,cmap="gray")
            #plt.show()
            #print(char)
            #try:


            char = 255*(char-1)
            pred = cf.predict(img = char, print_result=True)

            letter = 'Multi-letter/A'
            if(np.shape(pred)[0] == 1):
                letter = dic[np.argmax(pred)]

            print(c, r, letter)
            predictions.append((c, r, pred))
            #except:
            #    continue
            #showImage(cropped_characters[x], x)
        #print(predictions, '\n\n\n')
        out_dir = "transcripts/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + file + '.txt', 'w', encoding='utf-8') as out:
            out.write('predictions')



if __name__== "__main__":
  main()
