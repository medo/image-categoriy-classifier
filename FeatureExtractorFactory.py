from FeatureExtractor import FeatureExtractor

class FeatureExtractorFactory:

	@staticmethod
	def newInstance(image):
		return FeatureExtractor(image)