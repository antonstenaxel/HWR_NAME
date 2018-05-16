from baseline_cnn_classifier import Classifier
import sys

cf = Classifier(path_to_model = "Models/baseline_cnn.h5")

if(len(sys.argv) > 1):
    image_path = sys.argv[1]
else:
    print("Provide filepath to image as argument")

# Returns a 27 length array with probabilities correspongding to cf.dic
# Accepts both images as file paths and numpy matrices
cf.predict(img = image_path, print_result=True)
