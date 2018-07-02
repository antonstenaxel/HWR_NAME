# Team "HWR_NAME"
## Handwriting recognition for Dead Sea Scrolls

Project was done by **N**. Ahmadi, **A**. Karlsson, **M**d. Ataur Rahman, **E**. Kuzmenko


### Usage

The project should be run on Python 3.

* Clone the repo
* Install the requirements: 'pip install -r /path/to/requirements.txt'
* Run the project using: 'python Main.py /path/to/images'

The results are stored in the created folder 'transcripts'. 
It can optionally save in separate folders the steps of preprocessing: binarized image and segmented characters. 
The system can also output the edit distance measure if true transcripts are provided with the images (in the same directory).

Additional Notes: Please note that this repo has been used throughout the project and isn't really cleaned up to include only the final versions of the program, there are files that were used early on and discarded but kept for reference purposes, among them the Classification_Data folder. This is not the dataset that was used to train the network. 
