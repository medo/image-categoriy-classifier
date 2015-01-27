import numpy
import cv2 as cv

class FeatureExtractor:
  
  def __init__(self, image):
    self.image = image

  def extract_feature_vector(self):
    img = cv.imread(self.image.get_path())
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT()
    kp, des = sift.detectAndCompute(gray, None)
    return des
    

    