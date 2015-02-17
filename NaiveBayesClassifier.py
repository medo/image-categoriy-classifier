from Classifier import Classifier
import cv2
import numpy as np
import jsonpickle as jp
import sys, traceback

class NaiveBayesClassifier(Classifier):
    # Implementation of naive bayes classifier
    
    def __init__(self):
        self.classifier = cv2.NormalBayesClassifier()
        self.setErrorCount(0)
    
#     def save(self):
#         try:
#             #file name to be added to the configuration file
#             file = open("naiveBayesClassifier.json", 'w')
#             jsonData = jp.encode(self.classifier)
#             file.write(jsonData)
#             file.close()
#         except Exception, Argument:
#             print "Exception happened: ", Argument 
# 
#     def load(self):
#         try:
#             #file name to be added to the configuration file
#             file = open("naiveBayesClassifier.json")
#             jsonData = file.read()
#             self.classifier = jp.decode(jsonData)
#             file.close()
#         except Exception, Argument:
#             print "Exception happened: ", Argument
#             
    def save(self, outputFile):
        try:
            file = open(outputFile, 'w')
            jsonData = jp.encode(self.classifier)
            file.write(jsonData)
            file.close()
        except Exception, Argument:
            print "Exception happened: ", Argument 
        
    def load(self, outputFile):
        try:
            file = open(outputFile)
            jsonData = file.read()
            self.classifier = jp.decode(jsonData)
            file.close()
        except Exception, Argument:
            print "Exception happened: ", Argument

    
    def train(self, allFlag=False):
        try:
            return self.classifier.train(self.trainingData, self.trainingLabels, update = allFlag)
        except Exception, Argument:
            print "Exception happened: ", Argument
            traceback.print_stack()
        
    def predict(self, testData):
        try:
            response, results = classifier.predict(testData)
            return response
        except Exception, Argument:
            print "Exception happened: ", Argument
        