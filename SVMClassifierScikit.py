from Classifier import Classifier
import traceback
from sklearn import svm
from sklearn.externals import joblib

class SVMClassifierScikit(Classifier):

	def __init__(self):
		self.classifier = svm.LinearSVC()
		self.setErrorCount(0)
		self.setEvaluationsCount(0)

	def train(self):
		try:
			return self.classifier.fit(self.trainingData, self.trainingLabels)
		except Exception, Argument:
			print "Exception happened: ", Argument
			traceback.print_stack()

	def predict(self, testData):
		try:
			response = self.classifier.predict(testData)
			return response[0]
		except Exception, Argument:
			print "Exception happened: ", Argument

	def load (self,inputFile):
		try:
			self.classifier = joblib.load(inputFile) 

		except Exception, Argument:
			print "Exception happened: ", Argument
			traceback.print_stack()

	def save(self, outputFile):
		try:
			print("Writing Classifier to file %s" % outputFile)
			joblib.dump(self.classifier,outputFile,3)

		except Exception, Argument:
			print "Exception happened: ", Argument 
			traceback.print_stack()

	def calculateScore(self, testData):
		try:
			score = self.classifier.decision_function(testData)
			return score
		except Exception, Argument:
			print "Exception happened: ", Argument

