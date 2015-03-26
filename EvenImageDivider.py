import numpy as np

class EvenImageDivider:
	def __init__(self, image,n):
		self.image = image
		self.n = n

	def halvenAxis(self,axisLength,n):
		quotient = axisLength/n
		i=1
		remainder = axisLength % n
		result=[]
		result.append(quotient + remainder)
		for x in xrange(0,n-1):
			result.append((i+1)*quotient+remainder)
			i+=1
		return [0] + result

	def divider(self,sector):
		rows = len(self.image)
		cols= len(self.image[0])
		rowSplit=self.halvenAxis(rows,2)
		colSplit=self.halvenAxis(cols,self.n/2)
		result=[]
		counter=1
		for i in xrange(0,len(rowSplit)-1):
			for j in xrange(0,len(colSplit)-1):
				if sector==counter:
					return self.image[rowSplit[i]:rowSplit[i+1],colSplit[j]:colSplit[j+1]]
					print result
				else:
					counter+=1