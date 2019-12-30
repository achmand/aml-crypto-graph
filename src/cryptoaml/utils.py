class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)