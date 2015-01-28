from Classifier import Classifier
import cv2
import numpy as np
import jsonpickle

class NaiveBayesClassifier(Classifier):
    # Implementation of naive bayes classifier
    
    def __init__(self):
        self.classifier = cv2.NormalBayesClassifier()
        self.setErrorCount(0)
    
    def save(self):
        pass

    def load(self):
        pass
        
    def train(self, allFlag=False):
        return self.classifier.train(trainingDataMat, trainLabelsMat, update = allFlag)
        
    def predict(self, testData):
        response, results = classifier.predict(testData)
        return response