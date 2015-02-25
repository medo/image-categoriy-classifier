from math import sqrt
import numpy as np
import sys

class HistogramCalculator:

	def __init__(self, vocab):
		self.vocab = vocab

	def hist(self, sift):
		minVal = float("Inf")
		minIndex = -1
		count = 0.0
		length = len(sift)
		histogram = np.zeros(len(self.vocab))
		for point1 in sift:
			count += 1
			sys.stdout.write("\r%d%%" % ((count / length) * 100))
  			sys.stdout.flush()
			for index in range(0, len(self.vocab)):
				point2 = self.vocab[index]
				distance = 0
				distance = sqrt(np.sum(np.power((point2 - point1), 2)))
				if distance < minVal:
					minVal = distance
					minIndex = index
			histogram[minIndex] += 1			
		return histogram




