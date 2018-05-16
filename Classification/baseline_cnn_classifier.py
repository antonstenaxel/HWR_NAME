import os, sys
import cv2
import numpy as np
from keras.models import load_model

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
    26: 'Zayin'}


    def __init__(self,path_to_model):
        self.model = load_model(path_to_model)

    def reshape_image(self,img,size):
        shape = np.shape(img)
        new_shape = (shape/np.max(shape)*size).astype(int)
        img = cv2.resize(img, dsize=tuple(new_shape), interpolation=cv2.INTER_AREA)

        delta_h = size - new_shape[1]
        delta_w = size - new_shape[0]

        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = 255

        new_img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)

        return new_img

    def preprocess_image(self,img):
        new_image = self.reshape_image(img,28)
        new_image = new_image.reshape(1, 28, 28, 1)
        new_image /= 255

        return new_image

    def read_image(self,path):
        img = np.double(cv2.imread(path,0))
        return img

    def predict(self,img, print_result = True):

        if(type(img) == str):
            img = self.read_image(img)
            
        new_image = self.preprocess_image(img)
        p = self.model.predict_proba(new_image)

        if(print_result):
            cl = 0;
            for p_i in p[0]:
                print(" {:.4f} \t{} ".format(p_i,self.dic[cl]))
                cl += 1
        return p
