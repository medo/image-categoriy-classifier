import numpy as np
import os
from CategoriesManager import CategoriesManager
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
from collections import OrderedDict

# A tuple (x,y) where x is a binary label and y a confidence score

class AveragePrecisionCalculator:
	def __init__(self,path,dictionary_file):
		self.binary_labels = None
		self.path = path
		self.dictionary_file = dictionary_file
		self.scoreList = None
		self.tuplesList = None
		self.specificTuplesList = None
		self.dictOfevalCatNames = OrderedDict()
		self.listOfevalLabels = []
		self.listOfAP = []

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

	def generate_score_list(self,confidenceScore):
		if self.scoreList == None:
			self.scoreList = np.array(confidenceScore)
		else:
			self.scoreList = np.insert(self.scoreList,self.scoreList.size,confidenceScore)

	def get_score_list(self):
		return self.scoreList

	def generate_tuples_list(self):
		for i in range(0,len(self.binary_labels)):
			if self.tuplesList == None:
				self.tuplesList = zip(self.binary_labels[i],self.scoreList)
				self.sort_list_of_tuples_descendingly()
			else:
				tupledList = zip(self.binary_labels[i],self.scoreList)
				self.sort_given_list_of_tuples_descendingly(tupledList)
				self.tuplesList = np.vstack((self.tuplesList,tupledList))

	def get_tuples_list(self):
		return self.tuplesList

	def sort_list_of_tuples_descendingly(self):
		self.tuplesList.sort(reverse=True,key=lambda tup: tup[1])

	def sort_given_list_of_tuples_descendingly(self,listOfTuples):
		return listOfTuples.sort(reverse=True,key=lambda tup: tup[1])

	def split_tuples_list_per_class(self):
		self.load_category_dictionary()
		self.specificTuplesList = np.vsplit(self.tuplesList,classesHashtable.getCategoriesCount())

	def get_specific_tuples_list(self,label):
		return self.specificTuplesList[label]

	def extract_y_true_from_specific_tuples_list(self,specificTuplesList):
		return specificTuplesList[:,0]

	def extract_score_from_specific_tuples_list(self,specificTuplesList):
		return specificTuplesList[:,1]

	def calculate_average_precision_score(self, y_true, score):
		self.listOfAP.append(average_precision_score(y_true,score))
		return average_precision_score(y_true,score)

	def add_evaluated_category_name(self,categoryName):
		label = self.check_label_existence(categoryName)
		if not label == None:
			self.dictOfevalCatNames[label] = categoryName
			return

	def get_evaluated_labels(self):
		return self.dictOfevalCatNames.keys()

	def get_evaluated_category_names(self):
		return self.dictOfevalCatNames.values()

	def get_evaluated_categories_count(self):
		return len(self.dictOfevalCatNames)

	def calculate_map(self):
		return sum(self.listOfAP)/len(self.listOfAP)
