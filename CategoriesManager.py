import pickle
import sys, traceback

class CategoriesManager:
    
    def __init__(self):
        self.classes = {}
        
    def addClass(self, number, name):
        if self.classes.has_key(name.lower()) or self.classes.has_key(number):
            return False
        
        self.classes[name.lower()] = number
        self.classes[number] = name.lower()
        
        return True
    
    def getKeyValue(self, key):
        return self.classes.get(key)
    
    def getClassNumber(self, name):
        return self.getKeyValue(name)
    
    def getClassName(self, number):
        return self.getKeyValue(number)
    
    def saveToFile(self, outputFile):
        try:
            file = open(outputFile, 'wb')
            print("Writing Classes to file %s" % outputFile)
            pickle.dump(self.classes, file)
            file.close()
        except Exception, Argument:
            print "Exception happened: ", Argument     
    
    def loadFromFile(self, inputFile):
        try:
            file = open(inputFile, 'rb')
            self.classes = pickle.load(file)
        except Exception, Argument:
            print "Exception happened: ", Argument