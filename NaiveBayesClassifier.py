from Classifier import Classifier
import cv2
import numpy as np
import jsonpickle as jp

class NaiveBayesClassifier(Classifier):
    # Implementation of naive bayes classifier
    
    def __init__(self):
        self.classifier = cv2.NormalBayesClassifier()
        self.setErrorCount(0)
    
    def save(self):
        #file name to be added to the configuration file
        file = open("naiveBayesClassifier.json", 'w')
        jsonData = jp.encode(self.classifier)
        file.write(jsonData)
        file.close()

    def load(self):
        #file name to be added to the configuration file
        file = open("naiveBayesClassifier.json")
        jsonData = file.read()
        self.classifier = jp.decode(jsonData)
        file.close()
        
    def train(self, allFlag=False):
        return self.classifier.train(trainingDataMat, trainLabelsMat, update = allFlag)
        
    def predict(self, testData):
        response, results = classifier.predict(testData)
        return response