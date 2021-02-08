import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


class MyVector(object):
	def __init__(self, size, is_col=True, fill=0, init_values=None):
		self.v = []
		self.size = size
		self.is_col = is_col

		if(init_values == None):
			for i in range(size):
				self.v.append(fill)

		else:
			for i in range(size):
				self.v.append(init_values[i % len(init_values)])



	def __add__(self,w): # w is vector?
		w = self.CheckErr(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] + w.v[i])
			
		return(MyVector(self.size, True, fill=0, init_values=ans))


	def CheckErr(self, err):
		if(self.is_col != err.is_col):
			raise ValueError("both vectors must be column/row vectors")
		if(self.size != err.size):
			raise ValueError("vectors must be of same size")
		return err
		
	def __str__(self):
		s = "["
		if (self.is_col == False):
			for i in range(len(self.v)):
				s += str(self.v[i]) + " ,"
			s += "]"
		else:
			for i in range(len(self.v)):
				s += str(self.v[i]) + "" "\n"
			s += "]" 

		return s



#print(MyVector(4,fill=7))
#print(MyVector(5,is_col=False,init_values=[1,2,3]))
#print(MyVector(4,init_values=[4,1,2,4]) + MyVector(4,fill=7))


try: 
	print(MyVector(3,fill=1) + MyVector(4,fill=2))
except Exception as err1:
	print("Exception:", err1)
	try: 
		print(MyVector(3,is_col=False,fill=1) + MyVector(3,fill=2))
	except Exception as err2:
		print("Exception:", err2)
		print(MyVector(3,fill=15) + MyVector(3,fill=21))


