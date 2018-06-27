import os, sys
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utility_functions

class Classifier:
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


    def __init__(self,path_to_model):
        self.model = load_model(path_to_model)


    def preprocess_image(self,img,multi_letter):

        if(multi_letter == False):
            preprocessed_image = utility_functions.preprocess_single_letter(img)
        else:
            preprocessed_image = utility_functions.preprocess_multi_letter(img)

        #thinned_image = utility_functions.zhangSuen(preprocessed_image)

        #Data obtained from training model, only for thinned and augmented images
        feature_mean =  0.18548644

        feature_std = 0.35560286

        final_image = (preprocessed_image-feature_mean)/feature_std

        return final_image

    def multi_letter_predict(self,img,print_result = False, stride = 1):

        img_height, img_width = np.shape(img)

        p = np.zeros([len(self.dic),(img_width-28)//stride])
        x = img_width
        i = 0
        while(x > 28):

            window = img[:,x-28:x]
            p_i = self.single_letter_predict(window.reshape(1,28,28,1),False)
            p[:,i] = p_i

            x -= stride
            i += 1

        if(print_result):
            dim = np.shape(p)
            fig, ax = plt.subplots()
            im = ax.imshow(p,cmap="Blues")

            # We want to show all ticks...
            ax.set_xticks(np.arange(i))
            ax.set_yticks(np.arange(dim[0]))
            # ... and label them with the respective list entries
            ax.set_yticklabels([letter for letter in self.dic.values()])
            ax.set_xticklabels(range(i))

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")
            plt.grid(alpha = 0.2)
            ax.set_title("Probability Map (from right to left)")
            fig.tight_layout()
            plt.xlabel("Window-step")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

            #plot = plt.show()


        return p

    def single_letter_predict(self,img,print_result = False):

        p = self.model.predict_proba(img)

        if(print_result):
            cl = 0;
            for p_i in p[0]:
                print(" {:.4f} \t{} ".format(p_i,self.dic[cl]))
                cl += 1
        return p

    # Returns soft max array for single letter predictions and a
    # matrix of windows for a sliding window if multi letter.
    def predict(self,img, print_result = True):

        if(type(img) == str):
            img = cv2.imread(img,0)

        preprocessed_image = self.preprocess_image(img.copy(),multi_letter = False)


        p = self.single_letter_predict(preprocessed_image.reshape([1,28,28,1]), print_result = False)
        #pred = np.argmax(p)
        #plt.imshow(preprocessed_image,cmap="gray")
        #plt.title(self.label2char[self.class2label[pred]])
        #plt.show()
        if(p[0,27] > 0.8):
            preprocessed_multi_letter = self.preprocess_image(img.copy(), multi_letter = True)
            s = np.shape(preprocessed_multi_letter)
            if(s[1] < 28):
                p = np.zeros([1,29])
                p[0,28] = 1
            else:
               p = self.multi_letter_predict(preprocessed_multi_letter)

        return p
