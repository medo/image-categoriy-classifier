import cv2
import numpy as np
import jsonpickle
import sys, traceback

#from Classifier import Classifier
#from NaiveBayesClassifier import NaiveBayesClassifier
from ClassifierFactory import ClassifierFactory




#preparing the training data
trainingDataMat	= np.matrix(np.zeros(shape = (80, 2))).astype('float32')

trainLabels = np.zeros(80, dtype = np.float32)
print "trainLabelsInfo"
print type(trainLabels)

for index in range(40):
    
    #class 1
    trainingDataMat[index, 0] = np.float32(np.random.normal())
    trainingDataMat[index, 1] = np.float32(np.random.normal())
    
    #class 2
    trainingDataMat[index+40, 0] = 1.0 + np.float32(np.random.normal())
    trainingDataMat[index+40, 1] = np.float32(np.random.normal())
    
    trainLabels[index+40] = 1

trainLabelsMat = np.matrix(trainLabels)
print "trainLabelsMatInfo"
print type(trainLabelsMat)

print "trainDataMatInfo"
print type(trainingDataMat)

print 'Training Data'
print trainingDataMat
print 'Training Labels'
print trainLabelsMat

try:
    classifier = ClassifierFactory.createClassifier()
    classifier.setTrainingData(trainingDataMat) #param = matrix of the training data
    classifier.setTrainingLabels(trainLabelsMat) #param = matrix of the training labels
    classifier.train()
    classifier.save("naiveBayesClassifier.json")

except:
    traceback.print_stack()