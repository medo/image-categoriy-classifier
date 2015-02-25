#!/usr/bin/python

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import getopt, sys, os, traceback
import cv2

from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster
from SIFTManager import SIFTManager
from HistogramCalculator import HistogramCalculator
from ClassifierFactory import ClassifierFactory
from CategoriesManager import CategoriesManager

# Private helper functions

def __check_dir_condition(path):
    if not os.path.isdir(path):
        print("%s: No such directory" % (path)) 
        sys.exit(2)
        
def __check_file_condition(file):
    if not os.path.isfile(file):
        print("%s: No such file" % (file)) 
        sys.exit(2)
        
def __check_label_existence(label_name):
    label_number = classesHashtable.getClassNumber(str(label_name))
    if label_number == None:
        print ("Label %s is not trained in our database" % label_name)
    return label_number
    
def __from_array_to_matrix(array_data):
    return np.matrix(array_data).astype('float32')

def __get_image_features(img_file):
    extractor = FeatureExtractorFactory.newInstance(Image.from_local_directory(img_file))
    return extractor.extract_feature_vector()
                
def __init_histogram_calculator(vocab_file):
    print ("Loading vocabulary from: %s" % vocab_file)
    vocab = SIFTManager.load_from_local_file(vocab_file)
    global histCalculator
    histCalculator = HistogramCalculator(vocab)

def __load_classifier(classifier_file):
    print ("Loading classifier from: %s" % classifier_file)
    global classifier 
    classifier = ClassifierFactory.createClassifier()
    classifier.load(classifier_file)

def __load_category_dictionary(dictionary_file):
    print ("Loading Classes Hashtable from: %s" % dictionary_file)
    global classesHashtable
    classesHashtable = CategoriesManager()
    classesHashtable.loadFromFile(dictionary_file)

def __create_and_train_classifier():
    global classifier
    classifier = ClassifierFactory.createClassifier()
    classifier.setTrainingData(trainingDataMat)
    classifier.setTrainingLabels(trainingLabelsMat)
    classifier.train()

def __save_classifier(output_file):
    print ("Saving Classifier in: %s" % output_file)
    classifier.save(output_file)

def __save_categories_dictionary(output_file):
    print ("Saving Dictionary in: %s" % output_file)        
    classesHashtable.saveToFile(output_file)

# Main functions

def vocabulary(path, output_file):
    __check_dir_condition(path)
    
    count = 0
    cluster = KMeanCluster(100)
    for i in os.listdir(path):
        if i.endswith(".jpg") or i.endswith(".png"):
            try:
                print i
                count += 1
                imgfile = "%s/%s" % (path, i)
                vector = __get_image_features(imgfile)
                cluster.add_to_cluster(vector)
            except Exception, Argument:
                print "Exception happened: ", Argument 

    if count == 0:
        print ("%s contains no png/jpg images" % (path))
        return

    result = cluster.cluster()
    SIFTManager.save_to_local_file(result, output_file)


def evaluating(path, vocab_file, classifier_file, dictionary_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    __check_file_condition(classifier_file)
    __check_file_condition(dictionary_file)
    
    __init_histogram_calculator(vocab_file)  
    __load_classifier(classifier_file)    
    __load_category_dictionary(dictionary_file)
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("Evaluating label '%s'" % d)
            wrongPredictions = 0
            totalPredictions = 0
            label = __check_label_existence(d)

            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        vector = __get_image_features(imgfile)
                        bow = histCalculator.hist(vector)
                        bow = __from_array_to_matrix(bow)
                        totalPredictions += 1
                        correctResponse = classifier.evaluateData(bow, label)
                        if not correctResponse:
                            wrongPredictions += 1
                        
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
    
    print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))
    

def training(path, output_file, vocab_file, dictionary_output_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    
    __init_histogram_calculator(vocab_file)
    
    label = 0
    labelsVector = None
    bowVector = None
    global classesHashtable 
    classesHashtable = CategoriesManager()
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("Training label '%s'" % d)
            classesHashtable.addClass(label, d)
            correctLabel = classesHashtable.getClassNumber(d)
            
            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        vector = __get_image_features(imgfile)
                        bow = histCalculator.hist(vector)

                        if bowVector == None:
                            bowVector = bow
                        else:
                            bowVector = np.vstack((bowVector, bow))
                        if labelsVector == None:
                            labelsVector = np.array(correctLabel)
                        else:
                            labelsVector = np.insert(labelsVector, labelsVector.size, correctLabel)

                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
            if label == correctLabel:
                label += 1
    try:
        print "Training Classifier"
        
        global trainingDataMat
        trainingDataMat = __from_array_to_matrix(bowVector) 
        global trainingLabelsMat
        trainingLabelsMat = __from_array_to_matrix(labelsVector)

        print ("trainingDataMat", trainingDataMat)
        print ("trainingLabelsMat", trainingLabelsMat)
               
        __create_and_train_classifier()   
        __save_classifier(output_file)
        __save_categories_dictionary(dictionary_output_file)

    except Exception, Argument:
        print "Exception happened: ", Argument
        traceback.print_stack()


def evaluating2(path, vocab_file, classifier_file, dictionary_file):
    __check_dir_condition(path)
    __check_file_condition(vocab_file)
    __check_file_condition(classifier_file)
    __check_file_condition(dictionary_file)
    
    __init_histogram_calculator(vocab_file)  
    __load_classifier(classifier_file)    
    __load_category_dictionary(dictionary_file)
    
    yc_true = None
    yc_scores = None
    ym_true = None
    ym_scores = None
    ya_true = None
    ya_scores = None
    
    for d in os.listdir(path):
        subdir = ("%s/%s" % (path, d))
        if os.path.isdir(subdir):
            print ("Evaluating label '%s'" % d)
#             wrongPredictions = 0
#             totalPredictions = 0
            label = __check_label_existence(d)

            for f in os.listdir(subdir):
                if f.endswith(".jpg") or f.endswith(".png"):
                    try:
                        print f
                        imgfile = "%s/%s" % (subdir, f)
                        vector = __get_image_features(imgfile)
                        bow = histCalculator.hist(vector)
                        bow = __from_array_to_matrix(bow)
#                         totalPredictions += 1
#                         correctResponse = classifier.evaluateData(bow, label)
                        res = classifier.predict(bow)
#                         if not correctResponse:
#                             wrongPredictions += 1
                        label = int(round(label))
                        res = int(round(res))
                        
                        
                        if res == 0:
                            if yc_scores == None:
                                yc_scores = np.array(1)
                                ya_scores = np.array(0)
                                ym_scores = np.array(0)
                            else:
                                yc_scores = np.insert(yc_scores, yc_scores.size, 1)
                                ya_scores = np.insert(ya_scores, ya_scores.size, 0)
                                ym_scores = np.insert(ym_scores, ym_scores.size, 0)
                        elif res == 2:
                            if yc_scores == None:
                                yc_scores = np.array(0)
                                ya_scores = np.array(1)
                                ym_scores = np.array(0)
                            else:
                                yc_scores = np.insert(yc_scores, yc_scores.size, 0)
                                ya_scores = np.insert(ya_scores, ya_scores.size, 1)
                                ym_scores = np.insert(ym_scores, ym_scores.size, 0)
                            
                        else:
                            if yc_scores == None:
                                yc_scores = np.array(0)
                                ya_scores = np.array(0)
                                ym_scores = np.array(1)
                            else:
                                yc_scores = np.insert(yc_scores, yc_scores.size, 0)
                                ya_scores = np.insert(ya_scores, ya_scores.size, 0)
                                ym_scores = np.insert(ym_scores, ym_scores.size, 1)
                            
                        
                        if label == 0:
                            if yc_true == None:
                                yc_true = np.array(1)
                                ya_true = np.array(0)
                                ym_true = np.array(0)
                            else:
                                yc_true = np.insert(yc_true, yc_true.size, 1)
                                ya_true = np.insert(ya_true, ya_true.size, 0)
                                ym_true = np.insert(ym_true, ym_true.size, 0)
                                
                        elif label == 1:
                            if yc_true == None:
                                yc_true = np.array(0)
                                ya_true = np.array(0)
                                ym_true = np.array(1)
                            else:
                                yc_true = np.insert(yc_true, yc_true.size, 0)
                                ya_true = np.insert(ya_true, ya_true.size, 0)
                                ym_true = np.insert(ym_true, ym_true.size, 1)
                            
                        
                        else:
                            if yc_true == None:
                                yc_true = np.array(0)
                                ya_true = np.array(1)
                                ym_true = np.array(0)
                            else:
                                yc_true = np.insert(yc_true, yc_true.size, 0)
                                ya_true = np.insert(ya_true, ya_true.size, 1)
                                ym_true = np.insert(ym_true, ym_true.size, 0)
                        
                    except Exception, Argument:
                        print "Exception happened: ", Argument
                        traceback.print_stack()
            
#             print ("Label %s results:\n%d were wrongly predicted from %d" % (d, wrongPredictions, totalPredictions))
    
#     print ("Final results:\n%d were wrongly predicted from %d" % (classifier.getErrorCount(), classifier.getEvaluationsCount()))
#     y_true = y_true.astype('int')
#     print "y_true: " + str(y_true) + " " + str(type(y_true))
#     print "y_scores: " + str(y_scores) + " " + str(type(y_scores))
#      print "The average precision score: " + str(average_precision_score(label_binarize(y_true, classes=[0, 1, 2]), label_binarize(y_scores, classes=[0, 1, 2])))
    print "cow precision: "
    print average_precision_score(label_binarize(yc_true, classes=[0,1]), label_binarize(yc_scores, classes=[0,1]))
    print average_precision_score(yc_true, yc_scores) 
    print "motorbike precision: "
    print average_precision_score(label_binarize(ym_true, classes=[0,1]), label_binarize(ym_scores, classes=[0,1]))
    print average_precision_score(ym_true, ym_scores)
    print "car precision: "
    print average_precision_score(label_binarize(ya_true, classes=[0,1]), label_binarize(ya_scores, classes=[0,1]))
    print average_precision_score(ya_true, ya_scores)
    

def listAll(temp_file):
    classesHashtable = CategoriesManager()
    classesHashtable.loadFromFile(temp_file)
    classesHashtable.listCategories()


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
            
            if opt == '-d':
                listAll(arg)
                sys.exit()
            
    except getopt.GetoptError, e:
        print str(e)
        sys.exit(2)
	
	
if __name__ == "__main__":
    main(sys.argv[1:])
