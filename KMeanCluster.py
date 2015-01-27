import numpy as np
import cv2 as cv

class KMeanCluster:

	def __init__(self, clusters_count):
		self.clusters_count = clusters_count
		self.vector = None 
		
	def add_to_cluster(self, vector):
		if self.vector == None:
			self.vector = vector
		else:		
			self.vector = np.vstack((self.vector, vector))

	def cluster(self):
		criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		ret, labels, centers = cv.kmeans(self.vector, self.clusters_count, criteria, 10, 0)
		print centers
		return centers






