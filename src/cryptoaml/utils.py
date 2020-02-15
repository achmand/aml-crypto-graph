import os

class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def create_dir(dirName):
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory '" , dirName ,  "' created ")
    else:    
        print("Directory '" , dirName ,  "' already exists")