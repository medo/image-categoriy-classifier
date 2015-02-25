

class Image:
    
    @staticmethod
    def from_local_directory(path):
        return Image(path)

    
    def config_function(self, args):
        print "config function %s" % args


    def __preprocess(self, conf_file="config/preprocess.conf"):
        conf = open(conf_file, "r")
        line = conf.readline()
        while line:
            line = line.strip()
            if line[0] != '#':
                linecontents = line.split()
                if not hasattr(self, linecontents[0]):
                    print "Error in %s: %s is not defined in Image.py" % (conf_file, linecontents[0])
                    return
                function = getattr(self, linecontents[0])
                function(linecontents[1:])
            line = conf.readline()

    def __init__(self, path):
        self.path = path
        self.__preprocess()

    

   

    def get_path(self):
        return self.path

        
