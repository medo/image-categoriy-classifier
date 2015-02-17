#!/usr/bin/python

import numpy as np
from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
from ClassifierFactory import ClassifierFactory
import getopt, sys, os, traceback
import cv2
import jsonpickle

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
        print("%s: No such directory" % (path)) 
        sys.exit(2)
    if not os.path.isfile(vocab_file):
        print("%s: No such file" % (vocab_file)) 
        sys.exit(2)

    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    histCalculator = HistogramCalculator(vocab)
    
    label = 0 
    labelsVector = None
    bowVector = None
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            #label = d
            print ("Training label '%s'" % d)
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(imgfile))
                        vector = extractor.extract_feature_vector()
                        bow = histCalculator.hist(vector)
                        #print bow
                        
                        if bowVector == None:
                            bowVector = bow
                        else:
                            bowVector = np.vstack((bowVector, bow))
                        if labelsVector == None:
                            labelsVector = label
                        else:
                            labelsVector = np.vstack((labelsVector, label))
                        
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            label += 1
    try:
        print "Training Classifier"
        
        trainingDataMat = np.matrix(bowVector).astype('float32')
        trainingLabelsMat = np.matrix(labelsVector.ravel()).astype('float32')
        
        classifier = ClassifierFactory.createClassifier()
        classifier.setTrainingData(trainingDataMat)
        classifier.setTrainingLabels(trainingLabelsMat)
        classifier.train()
        classifier.save(output_file)
        
        print ("Saving Classifier in: %s" % output_file)
        
    except Exception, Argument:
        print "Exception happened: ", Argument
        traceback.print_stack()


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
