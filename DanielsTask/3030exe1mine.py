import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


class MyVector(object):
	def __init__(self, size, is_col=True, fill=0, init_values=None):
		self.v = []
		self.size = size
		self.is_col = is_col
		ind = 0

		if(init_values == None):
			for i in range(size):
				self.v.append(fill)

		else:
			for i in range(size):
				if(len(init_values) > ind):
					self.v.append(init_values[ind])
					ind = ind + 1
				else:
					self.v.append(init_values[0])
					ind = 1
		

	def __str__(self):
		s = "["
		if (self.is_col == False):
			for i in range(len(self.v)):
				s += str(self.v[i]) + ", "
			s += "]"
		else:
			for i in range(len(self.v)):
				s += str(self.v[i]) + "" "\n"
			s += "]"

		return s

# ---------------------------------------------------------------
# Check for validity of self and other. If scalar - will broadcast to a vector
# ---------------------------------------------------------------
	def __check_other(self, other):
		if not isinstance(other,MyVector):
			if (type(other) in [int, float]):
				other = MyVector(self.size, True, fill = other)
			else:
				raise ValueError("*** Wrong type of parameter")
			if (self.is_col == False or other.is_col == False):
				raise ValueError("*** both vectors must be column vectors")
		if (self.size != other.size):
			raise ValueError("*** vectors must be of same size")
		return other    

# ---------------------------------------------------------------
# ADD vectors
# ---------------------------------------------------------------
	def __add__(self,w):
		w = self.__check_other(w)
		res = []
		for i in range (self.size):
			res.append (self.vector[i] + w.vector[i])
		return (MyVector(self.size, True, fill = 0, init_values=res))



print(MyVector(4,fill=7))
print(MyVector(5,is_col=False,init_values=[1,2,3]))