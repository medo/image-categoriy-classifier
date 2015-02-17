import numpy as np

class SIFTManager:

    @staticmethod
    def load_from_local_file(file):
        try:
            file = open(file, 'r')
            data = np.load(file)
            return data
        except Exception, Argument:
            print "Exception happened: ", Argument

    @staticmethod
    def save_to_local_file(descriptors, filename):
        try:
            file = open(filename, 'w')
            print("Writing file to %s" % filename)
            np.save(file, descriptors)
            file.close()
        except Exception, Argument:
            print "Exception happened: ", Argument 