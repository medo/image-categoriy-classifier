from FeatureExtractorFactory import FeatureExtractorFactory
from Image import Image
from KMeanCluster import KMeanCluster


f = FeatureExtractorFactory.newInstance(Image.from_local_directory("images/sample.png"))
vector = f.extract_feature_vector()
clus = KMeanCluster(10)
clus.add_to_cluster(vector)
clus.cluster()