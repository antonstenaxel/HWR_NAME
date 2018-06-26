import os
import sys
sys.path.append('Classification')
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from Segmentation import Binarization, Remove_Calibration, Segmentation
from Classification.improved_classifier import Classifier
import json
import numpy as np

class2label = {0: 'Alef',
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

label2char = {'Alef':'א',
'Ayin':'ע',
'Bet':'ב',
'Dalet':'ד',
'Gimel':'ג',
'He':'ה',
'Het':'ח',
'Kaf':'כ',
'Kaf-final':'ך',
'Lamed':'ל',
'Mem':'ם',
'Mem-medial':'מ',
'Nun-final':'ן',
'Nun-medial':'נ',
'Pe':'פ',
'Pe-final':'ף',
'Qof':'ק',
'Resh':'ר',
'Samekh':'ס',
'Shin':'ש',
'Taw':'ת',
'Tet':'ט',
'Tsadi-final':'צ',
'Tsadi-medial':'צ',
'Waw':'ו',
'Yod':'י',
'Zayin':'ז',
'Multi-letter': '!',
'Noise': ''
}


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
    #Binarization.save_image(bin_image, file.split(".")[0])             

    # Returns a list of segmented characters and number of rows
    cropped_characters, row = Segmentation.segmentation(bin_image)
    #Segmentation.save_segmented_characters(cropped_characters, row, file)    

    return cropped_characters, row

def main():
    #cf = Classifier(path_to_model = "Classification/Models/baseline_cnn.h5")
    cf = Classifier(path_to_model = "Classification/Models/thinned_and_augmented_cnn_v2.h5")

    if(len(sys.argv) > 1):
        mypath = sys.argv[1]
    else:
        mypath = 'images/all_images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        print(file)
        file_name = mypath + os.sep + file

        cropped_characters, row = pre_processing(file_name, file)

        predictions = []
        for c, (char, r) in enumerate(zip(cropped_characters, row)):
            char = 255*(char-1)
            pred = cf.predict(img = char, print_result=True)

            #letter = 'Multi-letter/A'
            letter ='?'
            if(np.shape(pred)[0] == 1):
                letter = label2char[class2label[np.argmax(pred)]]
            #print(c, r, letter)
            predictions.append((c, r, letter))
            
        out_dir = "transcripts/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + file + '.txt', 'w', encoding='utf-8') as out:
            row = 0
            line = []
            for p in predictions:
                c, r, letter = p
                if r != row:
                    print(line)
                    out.write(''.join(line) + '\n')
                    row = r
                    line = []
                line.append(letter)


if __name__== "__main__":
  main()
