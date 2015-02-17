#!/usr/bin/python

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
import getopt, sys, os


def vocabulary(path, output_file):
    if not os.path.isdir(path):
        print("%s: no such directory" % (path)) 
        sys.exit(2)

    count = 0
    cluster = KMeanCluster(100)
    for i in os.listdir(path):
        if i.endswith(".jpg") or i.endswith(".png"):
            try:
                print i
                count += 1
                imgfile = "%s/%s" % (path, i)
                extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(imgfile))
                vector = extractor.extract_feature_vector()
                cluster.add_to_cluster(vector)
            except Exception, Argument:
                print "Exception happened: ", Argument 

    if count == 0:
        print ("%s contains no png/jpg images" % (path))
        return

    result = cluster.cluster()
    SIFTManager.save_to_local_file(result, output_file)


def training(path, output_file, vocab_file):
    if not os.path.isdir(path):
        print("%s: no such directory" % (path)) 
        sys.exit(2)

    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    histCalculator = HistogramCalculator(vocab)

    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            label = d
            print ("Training label '%s'" % label)
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(imgfile))
                        vector = extractor.extract_feature_vector()
                        bow = histCalculator.hist(vector)
                        print bow
                        #TODO: train the classifer with the label: 'label' and bag of words: 'bow'

                    except Exception, Argument:
                        print "Exception happened: ", Argument

    #TODO save the classifier to 'output_file'

def main(args):

    try:
        optlist, args = getopt.getopt(args, 'v:o:t:r:')
        optlist = dict(optlist)
        output_file = "vocab/vocab.sift"
        if "-o" in optlist:
            output_file = optlist["-o"]
        for opt, arg in optlist.iteritems():
            if opt == '-t':
                if "-r" not in optlist:
                    print "Usage: -t <training_dir> -r <reference_vocab>"
                    sys.exit(2)

                training(arg, output_file, optlist['-r'])
                sys.exit()
            if opt == '-v':
                vocabulary(arg, output_file)
                sys.exit()

    except getopt.GetoptError, e:

        print str(e)
        sys.exit(2)
	
	
if __name__ == "__main__":
    main(sys.argv[1:])
