import os
from pathlib import Path

class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def create_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)