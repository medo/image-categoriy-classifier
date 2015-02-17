
class Image:
	
    @staticmethod
    def from_local_directory(path):
        return Image(path)

    def __init__(self, path):
        self.path = path


    def get_path(self):
        return self.path

        