import os, sys
import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    def reshape_image(self,img,size,multi_letter = False):

        shape = np.shape(img)

        if(multi_letter):
            # If multiple letters, only reformat height
            new_shape = (shape/np.array(shape[0])*size).astype(int)
            new_img = cv2.resize(img, dsize=(new_shape[1],new_shape[0]), interpolation=cv2.INTER_AREA)

        else:
            # If single letter, reformat to square
            new_shape = (shape/np.max(shape)*size).astype(int)
            img = cv2.resize(img, dsize=(new_shape[1],new_shape[0]), interpolation=cv2.INTER_AREA)

            delta_h = size - new_shape[0]
            delta_w = size - new_shape[1]

            top, bottom = delta_h//2, delta_h-(delta_h//2)
            left, right = delta_w//2, delta_w-(delta_w//2)

            color = 255

            new_img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=color)

        return new_img

    def preprocess_image(self,img,multi_letter = False):
        shape = np.shape(img)
        new_image = img.reshape(1, shape[0], shape[1], 1)
        new_image /= 255

        return new_image

    def read_image(self,path):
        img = np.double(cv2.imread(path,0))
        return img

    def multi_letter_predict(self,img,print_result = True, stride = 1):
        img = self.reshape_image(img,28,True)
        img_height, img_width = np.shape(img)

        p = np.zeros([len(self.dic),img_width//stride])
        x = 0
        i = 0
        while(x < img_width):
            window = img[0:img_height,x:x+img_height]
            p_i = self.single_letter_predict(window,False)
            p[:,i] = p_i

            x += stride
            i += 1

        if(print_result):
            dim = np.shape(p)
            #
            # x = np.arange(0,dim[1])
            # y = np.arange(0,dim[0])
            # fig = plt.figure()
            # ax = fig.gca(projection='3d')
            #
            # X,Y = np.meshgrid(x,y)
            #
            # # Plot the surface.
            # surf = ax.plot_surface(X, Y, p, cmap=cm.coolwarm,linewidth=0, antialiased=False)
            #
            #
            # # Add a color bar which maps values to colors.
            # fig.colorbar(surf, shrink=0.5, aspect=5)
            #
            # plt.xlabel("Sliding window")
            # plt.ylabel("Letter")
            # plt.show()
            # plt.imshow(p,cmap='hot', interpolation='nearest')
            # plt.show()

            fig, ax = plt.subplots()
            im = ax.imshow(p,cmap="Blues")

            # We want to show all ticks...
            ax.set_xticks(np.arange(dim[1]))
            ax.set_yticks(np.arange(dim[0]))
            # ... and label them with the respective list entries
            ax.set_yticklabels([letter for letter in self.dic.values()])
            ax.set_xticklabels(range(dim[1]))

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

            ax.set_title("Probability Map")
            fig.tight_layout()
            plt.xlabel("Window-step")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            # plot = plt.show()





        return p

    def single_letter_predict(self,img,print_result = True):
        img = self.reshape_image(img,28, False)
        preprocessed_image = self.preprocess_image(img)
        p = self.model.predict_proba(preprocessed_image)

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
            img = self.read_image(img)

        img_height,img_width = np.shape(img)

        if(img_width/img_height > 1.2):
            p = self.multi_letter_predict(img,print_result)
        else:
            p = self.single_letter_predict(img, print_result)



        return p
