__author__ = 'mohammed'

#Python version of Lecture's code for naive bayes classifier

import cv2
import numpy as np
import jsonpickle

#preparing the training data
trainingDataMat	= np.matrix(np.zeros(shape = (80, 2))).astype('float32')

trainLabels = np.zeros(80, dtype = np.float32)

for index in range(40):
    
    #class 1
    trainingDataMat[index, 0] = np.float32(np.random.normal())
    trainingDataMat[index, 1] = np.float32(np.random.normal())
    
    #class 2
    trainingDataMat[index+40, 0] = 1.0 + np.float32(np.random.normal())
    trainingDataMat[index+40, 1] = np.float32(np.random.normal())
    
    trainLabels[index+40] = 1

trainLabelsMat = np.matrix(trainLabels)

classifier = cv2.NormalBayesClassifier()
temp = classifier.train(trainingDataMat, trainLabelsMat, update = False)
print temp

#preparing the test data
testDataMat	= np.matrix(np.zeros(shape = (20, 2))).astype('float32')

testLabels = np.zeros(20, dtype = np.float32)

for index in range(10):
    
    #class 1
    testDataMat[index, 0] = np.float32(np.random.normal())
    testDataMat[index, 1] = np.float32(np.random.normal())
    
    #class 2
    testDataMat[index+10, 0] = 1.0 + np.float32(np.random.normal())
    testDataMat[index+10, 1] = np.float32(np.random.normal())
    
    testLabels[index+10] = 1

testLabelsMat = np.matrix(testLabels)

response = 0.0
misClassificationCount = 0

length = len(testDataMat)

for index in range(length):
    
    response, results = classifier.predict(testDataMat[index])
    
    if np.absolute(response - testLabelsMat[0, index]) > 0.00001:
        misClassificationCount += 1
        


print "Error Rate : " + str(misClassificationCount)
