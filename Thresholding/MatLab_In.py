import os
from os import listdir
from os.path import isfile, join
import matlab.engine


def call_matlab_script(file_name):

    eng = matlab.engine.start_matlab()
    eng.test(nargout=0)


def main():

    mypath = 'images'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for file in onlyfiles:
        call_matlab_script(mypath+'\\'+file)

    # thresholding('sample1.jpg')

if __name__== "__main__":
  main()
