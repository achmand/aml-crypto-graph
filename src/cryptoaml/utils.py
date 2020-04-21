import pickle
from pathlib import Path

class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def create_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)

def read_pickle(path):
    pickle_object = None
    with open(path, "rb") as obj_file:
        pickle_object = pickle.load(obj_file)
    return pickle_object

def save_pickle(path, obj):
    with open(path, "wb") as obj_file:
        pickle.dump(obj, obj_file)