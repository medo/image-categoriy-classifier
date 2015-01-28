from NaiveBayesClassifier import NaiveBayesClassifier
from Classifier import Classifier
import cv2

class ClassifierFactory(object):
    # This is the class that is responsible for creating
    # a classifier based on its type
    
    @staticmethod
    def createClassifier(type):
        if type == 'naive':
            return NaiveBayesClassifier()
             