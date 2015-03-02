from FeatureExtractor import FeatureExtractor

class FeatureExtractorFactory:

    @staticmethod
    def newInstance(image,memory):
        return FeatureExtractor(image,memory)