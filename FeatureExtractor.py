import numpy
import cv2 as cv

class FeatureExtractor:
  
  def __init__(self, image):
    self.image = image

  def extract_feature_vector(self):
    img = cv.imread(self.image.get_path())
    img_resized = cv.resize(img, (300, 250)) #resizing the image
    gray = cv.cvtColor(img_resized, cv.COLOR_BGR2GRAY)
    gray_equalized = cv.equalizeHist(gray) #equalizing the histogram (must take a gray scale image)
#     clahe = cv.createCLAHE(tileGridSize=(10,10))
#     gray2 = clahe.apply(gray)
    sift = cv.SIFT()
    kp, des = sift.detectAndCompute(gray2, None)
    return des
    

    