import os, random, cv2,sys
import numpy as np
import matplotlib.pyplot as plt
path = "/Users/Karlsson/Documents/Skola/Handwriting_Recognition/monkbrill_171005/"

def get_letter(letter):
    file = random.choice(os.listdir(path+letter))
    img = np.double(cv2.imread(path+letter+"/"+file,0))
    img_cutout = cut_out_image(img)
    return img_cutout

def binarize(img):
    treshold = 100

    img[img<=treshold] = 1
    img[img>treshold] = 0
    return img.astype(np.uint8)

def cut_out_image(img):

    binarized_image = binarize(img)
    connectivity = 8
    output = cv2.connectedComponentsWithStats(binarized_image, connectivity, cv2.CV_32S)
    centroids = output[3]
    num_labels = output[0]
    labels = output[1]
    stats = output[2]
    # Biggest component
    biggest_component = np.argmax(stats[1:,cv2.CC_STAT_AREA])+1
    # Bounding box of biggest component
    left,top,width,height = stats[biggest_component,0:4]

    # Cut out from original image
    img*= (labels == biggest_component)
    new_img = img[top:top+height,left:left+width]

    new_centroid = centroids[biggest_component,:]-np.array([left,top])

    return new_img, new_centroid

def combine_letters(list_of_letters):

    n_letters = len(list_of_letters)
    letters = []
    centroids = []

    total_width = 0
    max_height = 0
    max_centroid = 0

    for letter in list_of_letters:
        img, centroid = get_letter(letter)
        height, width = np.shape(img)
        total_width += width
        if (height > max_height):
            max_height = height

        if(centroid[1] > max_centroid):
            max_centroid = centroid[1]

        letters += [img]
        centroids += [centroid]

    combined_image = np.zeros([max_height*2,total_width])

    current_x = 0
    max_h = 0
    for i in range(n_letters):
        img = letters[i]
        height,width = np.shape(img)
        height_offset = int(max_centroid-centroids[i][1])
        combined_image[height_offset:height_offset+height,current_x:current_x+width] = img
        current_x += width

        if(height_offset+height > max_h):
            max_h = height_offset+height


    return combined_image[:max_h,:]

#if __name__ == '__main__':
    #path = sys.argv[1]
    #letters = sys.argv[2:]
    #img = combine_letters(letters)
    #plt.imshow(img,cmap="binary")
    #plt.show()
