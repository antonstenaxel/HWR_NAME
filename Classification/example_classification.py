from baseline_cnn_classifier import Classifier
import sys

cf = Classifier("Models/baseline_cnn.h5")

if(len(sys.argv) > 1):
    image_path = sys.argv[1]
else:
    print("Provide filepath to image as argument")

cf.predict(image_path)
