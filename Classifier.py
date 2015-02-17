from abc import ABCMeta, abstractmethod
import cv2
import numpy as np
import jsonpickle as jp
import sys, traceback

class Classifier(object):
    # Abstract class representing classifier object
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def train(self, allFlag=False):
        pass
        
    @abstractmethod    
    def predict(self, testData):
        pass
    
    def predict(self):
        self.predict(self, self.getTestingData())
        
    def setTrainingData(self, trainData):
        self.trainingData = trainData

    def setTrainingLabels(self, trainLabels):
        self.trainingLabels = trainLabels

    def setTestingData(self, testData):
        self.testingData = testData

    def setErrorCount(self, newCount):
        self.errorCount = newCount        

    def incrementErrorCount(self):
        self.errorCount += 1

    def getTrainingData(self):
        return self.trainingData  
        
    def getTrainingLabels(self):
        return self.trainingLabels

    def getTestingData(self):
        return self.testingData  
        
    def getErrorCount(self):
        return self.errorCount
        
    def evaluateData(self, testData, correctResponse):
        response = self.predict(self, testData)
        
        if np.absolute(response - correctResponse) > 0.00001:
            self.incrementErrorCount()

   
   
   
   