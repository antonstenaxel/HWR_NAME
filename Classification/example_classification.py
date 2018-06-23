from improved_classifier import Classifier
import sys

cf = Classifier(path_to_model = "Models/thinned_and_augmented_cnn_v2.h5")

if(len(sys.argv) > 1):
    image_path = sys.argv[1]
else:
    image_path = "Test_Images/test_img.pgm"

# Returns a 27 length array with probabilities correspongding to cf.dic
# Accepts images as either file paths or numpy matrices
cf.predict(img = image_path, print_result=True)
