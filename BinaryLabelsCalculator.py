import numpy as np
import os
from CategoriesManager import CategoriesManager
from sklearn.preprocessing import label_binarize

class BinaryLabelsCalculator:
	def __init__(self,path,dictionary_file):
		self.binary_labels = None
		self.path = path
		self.dictionary_file = dictionary_file

	def get_binary_labels(self):
		return self.binary_labels

	def load_category_dictionary(self):
		global classesHashtable
		classesHashtable = CategoriesManager()
		classesHashtable.loadFromFile(self.dictionary_file)

	def check_label_existence(self,label_name):
		label_number = classesHashtable.getClassNumber(str(label_name))
		if label_number == None:
			print ("Label %s is not trained in our database" % label_name)
		return label_number

	def generate_binary_labels(self):
		self.load_category_dictionary()
		path = self.path
		y_true = None
		count = 0
		alle = None
		for d in os.listdir(path):
			subdir = ("%s/%s" % (path,d))
			if alle == None:
				alle = y_true
			else:
				alle = np.hstack((alle,y_true))
			if os.path.isdir(subdir):
				label = self.check_label_existence(d)
				
				if label == None:
					continue

				for f in os.listdir(subdir):
					if f.endswith(".jpg") or f.endswith(".png"):
						count +=1
				y_true = [label] * count
				count = 0
		alle = np.hstack((alle, y_true))
		allCategories = range(0,classesHashtable.getCategoriesCount())
		required = label_binarize(alle,allCategories)
		cols = len(required[0])
		for i in range (0,cols):
			if self.binary_labels == None:
				self.binary_labels = required[:,i]
			else:
				self.binary_labels = np.vstack((self.binary_labels, required[:,i]))
		return self.binary_labels

