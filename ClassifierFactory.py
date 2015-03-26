from NaiveBayesClassifier import NaiveBayesClassifier
from Classifier import Classifier
from SVMClassifier import SVMClassifier
import cv2

class ClassifierFactory(object):
    # This is the class that is responsible for creating
    # a classifier based on its type
    
    @staticmethod
    def createClassifier(type='svm'):
        if type == 'naive':
            return NaiveBayesClassifier()

        if type == 'svm':
        	return SVMClassifier()
             