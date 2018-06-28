import os
import sys
sys.path.append('Classification')
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from Segmentation import Binarization, Remove_Calibration, Segmentation
from Classification.improved_classifier import Classifier
from Ling.ngram_probs import gen_characters, train_char_lm, final_probs
import json
import numpy as np
import docx2txt
import Min_Edit

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
    cropped_characters, row, no_ofChar = Segmentation.segmentation(bin_image)
    # Segmentation.save_segmented_characters(cropped_characters, row, file)

    return cropped_characters, row, no_ofChar

def main():

    cf = Classifier(path_to_model = "Classification/Models/improved_augmented_cnn_v6.h5")
    characters = gen_characters('Ling/ngrams_frequencies.csv')
    data = ' '.join(characters)
    lm = train_char_lm(data, order=2)

    if(len(sys.argv) > 1):
        mypath = sys.argv[1]
    else:
        mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    gold_char = ''

    for file in onlyfiles:
        print("\n"+file)
        file_name = mypath + os.sep + file

        # Considering both the images and gold outputs will be in the same folder
        file_extension = file.split(".")[1]
        cropped_characters = []
        row = []

        if file_extension != 'docx':
            # segmented images, row number and how many characters in case of multiple character
            cropped_characters, row, no_ofChar = pre_processing(file_name, file)
        else:
            gold_text = docx2txt.process(file_name)

            for ch in gold_text:
                if ch.isalnum():
                    gold_char += ch

        predictions = []
        history = '  '
        pred_char = ''
        for c, (char, r) in enumerate(zip(cropped_characters, row)):
            char = 255*(char-1)

            pred = cf.predict(img = char, print_result=False)

            #letter = 'Multi-letter/A'
            letter ='?'
            if(np.shape(pred)[0] == 1):
                letter = label2char[class2label[np.argmax(pred)]]
            #print(c, r, letter)
            if letter == '?':
                try:
                    letter = final_probs(lm, history[-2:])
                except:
                    pass
            history += letter
            predictions.append((c, r, letter))

            # This is to calculate Levenshtein distance
            if letter != ' ':
                pred_char += letter


        out_dir = "transcripts/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + file + '.txt', 'w', encoding='utf-8') as out:
            row = 0
            line = []
            for p in predictions:
                c, r, letter = p
                if r != row:
                    #final_probs(lm, "א ")
                    out.write(''.join(line) + '\n')
                    row = r
                    line = []
                line.append(letter)

        # Calculate Levenshtein (Minimum Edit) distance between gold and predicted output (if you place a .docx in the input folder)
        if file_extension != 'docx' and len(gold_char):

            print("\nGold Output:")
            print(gold_char)
            print("\nPredicted Output:")
            print(pred_char)

            print("\nMinimum edit distance = ", Min_Edit.min_edit_distance(gold_char, pred_char))

            # Initializing gold_char so that it doesn't affect the images without gold labels
            gold_char = ''


if __name__== "__main__":
  main()
