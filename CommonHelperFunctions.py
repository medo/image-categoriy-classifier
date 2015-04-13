from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
import numpy as np
import getopt, sys, os, traceback
import cv2, csv

class CommonHelperFunctions:
	def __init__(self):
		pass

	def check_dir_condition(self,path):
		if not os.path.isdir(path):
			print("%s: No such directory" % (path)) 
			sys.exit(2)

	def check_file_condition(self,file):
		if not os.path.isfile(file):
			print("%s: No such file" % (file)) 
			sys.exit(2)

	def get_image_name_from_path(self,path):
		imgName = path.split("/")
		return imgName[len(imgName)-1]

	def load_image(self,img):
		return cv2.imread(img)

	def get_category_name_from_file_name(self,file_name):
		categoryName = file_name.split("_")
		categoryName = categoryName[1].split(".csv")[0]
		return categoryName

	def from_array_to_matrix(self,array_data):
		return np.matrix(array_data).astype('float32')

	def file_len(self,fname):
		with open(fname) as f:
			for i, l in enumerate(f):
				pass
		return i + 1

	def csv_references_at_least_one_image(self,path,csv_file_name):
		try:
			file_name = ("%s/%s" % (path, csv_file_name))
			with open(file_name) as fileReader:
				reader = csv.reader(fileReader, delimiter=' ')
				for j in reader:
					if os.path.isfile(file_name):
						return True
		except Exception, Argument:
			print "Exception happened: ", Argument
			return False

	def belongs_to_class(self,instance,correctClass):
		return instance.__class__.__name__ == correctClass


