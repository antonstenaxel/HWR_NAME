import os, math, random
import numpy as np
import cv2
import skimage.io as io
from skimage.filters import threshold_otsu

def pad_image(image, padding, value):
    height, width = np.shape(image)
    
    new_image = value*np.ones([height+2*padding, width+2*padding])
    
    new_image[padding:padding+height, padding:padding+width] = image.copy()
    
    return new_image.astype(np.uint8)

def preprocess_image(img):

    img = pad_image(img,3,255)

    #Binarize image
    binarized_image = img.copy()
    threshold = 127
    binarized_image[img<=threshold] = 1
    binarized_image[img>threshold] = 0
    
    #Filter out biggest component
    output = cv2.connectedComponentsWithStats(binarized_image, 4, cv2.CV_32S)
    labels = output[1]
    stats = output[2]
    
    biggest_components =  np.argsort(-stats[:,4])
    
    #if(stats[biggest_components[0],0] == 0 and stats[biggest_components[0],1] == 0 ):
    biggest_component = biggest_components[1]
    #else:
        #biggest_component = biggest_components[0]
    
    binarized_image[labels != biggest_component] = 0
    binarized_image[labels == biggest_component] = 1
    #Get bounding box of resulting image
    left,top,width,height = stats[biggest_component,0:4]

    x1 = left
    x2 = left + width
    y1 = top
    y2 = top + height

    boxed_image = binarized_image[y1:y2,x1:x2].astype(np.uint8)

    #Reshape, but keep ratio
    original_shape = np.shape(boxed_image)
    new_shape = (original_shape/np.max(original_shape)*25).astype(int)
    resized_image = cv2.resize(boxed_image,dsize = (new_shape[1],new_shape[0]))
    
    #Blur
    sigma = 0.5
    smooth_image = 255*cv2.GaussianBlur(resized_image.astype(np.float64),(3,3),sigmaX=sigma,sigmaY=sigma)
    
    #Zeropad to make it desired size
    height,width  = new_shape

    left_padding = (28-width)//2
    top_padding = (28-height)//2

    padded_image = np.zeros([28,28])
    padded_image[top_padding:top_padding+height, left_padding:left_padding+width] = smooth_image

    return padded_image.astype(np.uint8)


def neighbours(x,y,image):
    #"Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    #"No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    Img_Original = image.copy()
    Otsu_Threshold = threshold_otsu(image.copy())   
    BW_Original = Img_Original > Otsu_Threshold 
    
    #"the Zhang-Suen Thinning Algorithm"
    Image_Thinned = BW_Original  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0




    #Blur                                                                                 
    sigma = 0.6
    smooth_image = 255*cv2.GaussianBlur(Image_Thinned.astype(np.float64),(3,3),sigmaX=sigma,sigmaY=sigma)

    return smooth_image.astype(np.uint8)
 
