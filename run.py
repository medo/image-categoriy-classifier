#!/usr/bin/python

import numpy as np
import getopt, sys, os, traceback
import cv2
import jsonpickle

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
from ClassifierFactory import ClassifierFactory
from DictionaryManager import DictionaryManager

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


def evaluating(path, vocab_file, classifier_file, dictionary_file):
    if not os.path.isdir(path):
        print("%s: No such directory" % (path)) 
        sys.exit(2)
    if not os.path.isfile(vocab_file):
        print("%s: No such file" % (vocab_file)) 
        sys.exit(2)
    if not os.path.isfile(vocab_file):
        print("%s: No such file" % (classifier_file)) 
        sys.exit(2)
    if not os.path.isfile(vocab_file):
        print("%s: No such file" % (dictionary_file)) 
        sys.exit(2)

    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    histCalculator = HistogramCalculator(vocab)
    
    print ("Loading classifier from: %s" % classifier_file)
    classifier = ClassifierFactory.createClassifier()
    classifier.load(classifier_file) 
    print "Classifier"
    print classifier
    
    print ("Loading Classes Hashtable from: %s" % dictionary_file)
    classesHashtable = DictionaryManager()
    classesHashtable.loadFromFile(dictionary_file)
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("Evaluating label '%s'" % d)
            wrongPredictions = 0
            totalPredictions = 0
            label = classesHashtable.getClassNumber(str(d))
            if label == None:
                print ("Label %s is not trained in our database" % d)
 
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(imgfile))
                        vector = extractor.extract_feature_vector()
                        bow = histCalculator.hist(vector)
                        bow = np.matrix(bow).astype('float32')
                        print ("bow Type %s %s" % (type(bow), bow))
                        totalPredictions += 1
                        correctResponse = classifier.evaluateData(bow[0], label)
                        if not correctResponse:
                            wrongPredictions += 1
                            
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
    
    print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))
    

def training(path, output_file, vocab_file, dictionary_output_file):
    print ("in training %s" % dictionary_output_file)
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
    classesHashtable = DictionaryManager()
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            #label = d
            print ("Training label '%s'" % d)
            classesHashtable.addClass(label, d)
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
        print "Classifier"
        print classifier
        print ("Saving Classifier in: %s" % output_file)
        classifier.save(output_file)
            
        print ("Saving Dictionary in: %s" % dictionary_output_file)        
        classesHashtable.saveToFile(dictionary_output_file)

    except Exception, Argument:
        print "Exception happened: ", Argument
        traceback.print_stack()


def main(args):

    try:
        optlist, args = getopt.getopt(args, 'v:o:t:r:d:e:c:')
        optlist = dict(optlist)
        output_file = "vocab/vocab.sift"
        if "-o" in optlist:
            output_file = optlist["-o"]
        for opt, arg in optlist.iteritems():
            if opt == '-t':
                if "-r" not in optlist or "-d" not in optlist:
                    print "Usage: -t <training_dir> -r <reference_vocab> -d <dictionary_output>"
                    sys.exit(2)

                training(arg, output_file, optlist['-r'], optlist['-d'])
                sys.exit()
                
            if opt == '-v':
                vocabulary(arg, output_file)
                sys.exit()
                
            if opt == '-e':
                if "-r" not in optlist or "-c" not in optlist or "-d" not in optlist:
                    print "Usage: -e <evaluating_dir> -r <reference_vocab> -c <reference_classifier> -d <reference_dictionary>"
                    sys.exit(2)
                
                evaluating(arg, optlist['-r'], optlist['-c'], optlist['-d'])
                sys.exit()

    except getopt.GetoptError, e:
        print str(e)
        sys.exit(2)
	
	
if __name__ == "__main__":
    main(sys.argv[1:])
