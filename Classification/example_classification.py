from baseline_cnn_classifier import Classifier
import sys

cf = Classifier(path_to_model = "Models/baseline_cnn.h5")

if(len(sys.argv) > 1):
    image_path = sys.argv[1]
else:
    image_path = "Test_Images/29.jpg"

# Returns a 27 length array with probabilities correspongding to cf.dic
# Accepts images as either file paths or numpy matrices
cf.predict(img = image_path, print_result=True)
