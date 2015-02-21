from Classifier import Classifier
import cv2
import sys, traceback

class NaiveBayesClassifier(Classifier):
    """This class represents the implementation of a naive bayes classifier"""
    
    def __init__(self):
        """It represents the __init__ method of the NaiveBayesClassifier
        
        Attributes:
            classifier (NormalBayesClassifier): instance of cv2 implementation of Naive Bayes Classifier
        
        """
        self.classifier = cv2.NormalBayesClassifier()
        self.setErrorCount(0)
        self.setEvaluationsCount(0)
    
    def save(self, outputFile):
        """This method saves the classifier instance to an output file
        
        Args:
            outputFile (str): is the file where the classifier needed to be saved
        
        """
        try:
            print("Writing Classifier to file %s" % outputFile)
            self.classifier.save(outputFile)
            
        except Exception, Argument:
            print "Exception happened: ", Argument 
            traceback.print_stack()
            
    def load(self, inputFile):
        """This method loads the classifier instance from an input file
        
        Args:
            inputFile (str): is the file where the classifier will be loaded from
        
        """
        try:
            self.classifier.load(inputFile)
                      
        except Exception, Argument:
            print "Exception happened: ", Argument
            traceback.print_stack()
    
    def train(self, allFlag=False):
        """This method trains the classifier with a matrix of training data and their corresponding labels
        
        Args:
            allFlag (bool, optional): it is responsible for determining whether to start the training from scratch
                                      or continue on previous trained data, defaults to False.
        
        """
        try:
            return self.classifier.train(self.trainingData, self.trainingLabels, update = allFlag)
        except Exception, Argument:
            print "Exception happened: ", Argument
            traceback.print_stack()
        
    def predict(self, testData):
        """This method predicts a given the category of given test data using the classifier
        
        Args:
            testData (matrix): represents the matrix of the bag of words of the needed element to be predicted
        
        Returns:
            The number of the predicted Category
        
        """
        try:
            response, results = self.classifier.predict(testData)
            return response
        except Exception, Argument:
            print "Exception happened: ", Argument
        