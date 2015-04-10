from NaiveBayesClassifier import NaiveBayesClassifier
from Classifier import Classifier
from SVMClassifier import SVMClassifier
from SVMClassifierScikit import SVMClassifierScikit
import cv2

class ClassifierFactory(object):
    # This is the class that is responsible for creating
    # a classifier based on its type
    
    @staticmethod
    def createClassifier(type='svmscikit'):
        if type == 'naive':
            return NaiveBayesClassifier()

        if type == 'svm':
        	return SVMClassifier()

        if type == 'svmscikit':
        	return SVMClassifierScikit()
             