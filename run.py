#!/usr/bin/python

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
import getopt, sys, os


def vocabulary(path, output_file):
    if not os.path.isdir(path):
        print("%s: no such directory" % (path)) 
        return

    count = 0
    cluster = KMeanCluster(100)
    for i in os.listdir(path):
        if i.endswith(".jpg") or i.endswith(".png"):
            try:
                print i
                count += 1
                extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory("images/sample.png"))
                vector = extractor.extract_feature_vector()
                cluster.add_to_cluster(vector)
            except Exception, Argument:
                print "Exception happened: ", Argument 

    if count == 0:
        print ("%s contains no png/jpg images" % (path))
        return

    result = cluster.cluster()
    SIFTManager.save_to_local_file(result, output_file)



def main(args):

    try:
        optlist, args = getopt.getopt(args, 'v:o:')
        optlist = dict(optlist)
        output_file = "vocab/vocab.sift"
        for opt, arg in optlist.iteritems():
            if opt == '-v':
                if "-o" in optlist:
                    output_file = optlist["-o"]
                vocabulary(arg, output_file)
                sys.exit()

    except getopt.GetoptError, e:

        print str(e)
        sys.exit(2)
	
	



if __name__ == "__main__":
    main(sys.argv[1:])
