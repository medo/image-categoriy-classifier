import numpy as np
class BagOfWordsVectorCalculator:

	def __init__(self):
		self.bowVector = None
		self.mergedBowVector = None

	def createMergedBow(self,bow):
		if self.mergedBowVector == None:
			self.mergedBowVector = bow
		else:
			self.mergedBowVector = np.hstack((self.mergedBowVector,bow))

	def createBowVector(self,bow):
		if self.bowVector == None:
			self.bowVector = bow
		else:
			self.bowVector = np.vstack((self.bowVector, bow))

	def getMergedBow(self):
		return self.mergedBowVector

	def getBowVector(self):
		return self.bowVector

	def emptyMergedBow(self):
		self.mergedBowVector = None

	def emptyBowVector(self):
		self.bowVector = None