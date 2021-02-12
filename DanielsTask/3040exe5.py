import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10
import math


class MyVector(object):
	def __init__(self, size, is_col=True, fill=0, init_values=None):
		self.v = []
		self.size = size
		self.is_col = is_col
		#self.init_values = init_values

		if(init_values == None):
			for i in range(size):
				self.v.append(fill)

		else:
			for i in range(size):
				self.v.append(init_values[i % len(init_values)])


	def __add__(self,w): 
		w = self.__check_other(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] + w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))

	def __mul__(self,w):
		w = self.__check_other(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] * w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))


	def __truediv__(self,w):
		w = self.__check_other(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] / w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))

	def __sub__(self, w):
		w = self.__check_other(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] - w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))


	def __check_other(self, err):
		if (not isinstance(err, MyVector)):
			if(type(err) in [int, float]):
				err = MyVector(self.size, self.is_col, fill=err)
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



	def __getitem__(self, key):
		return self.v[key]


	def __setitem__(self, key,value):
		self.v[key] = value


	def __len__(self):
		return self.size
	

	def transpose(self):
		return(MyVector(self.size, not self.is_col, fill=0, init_values=self.v))


	def __radd__(self, other):
		return self + other

	def __rsub__(self, other):
		return self - other

	def __rmul__(self, other):
		return self * other

	def __rtruediv__(self, other):
		return self / other


	def __lt__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] < other.v[i]):
				res[i] = 1
		return res


	def __le__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] > other.v[i]):
				res[i] <= 1
		return res


	def __eq__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] == other.v[i]):
				res[i] = 1
		return res

	def __ne__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] != other.v[i]):
				res[i] = 1
		return res


	def __ge__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] >= other.v[i]):
				res[i] = 1
		return res


	def __gt__(self,other):
		other = self.__check_other(other)
		res = MyVector(self.size, fill=0)
		for i in range(self.size):
			if(self.v[i] > other.v[i]):
				res[i] = 1
		return res

	def dot(self,err):
		if (not isinstance(err, MyVector)):
			raise ValueError("wrong type of parameter")
		if(self.is_col == err.is_col):
			raise ValueError("both vectors are column/row vectors")
		if(self.size != err.size):
			raise ValueError("vectors must be of same size")

		sdot = 0
		if(self.is_col != err.is_col):
			for i in range(self.size):
				sdot += self.v[i] * err.v[i]
		return sdot


	def norm(self):
		normm = 0
		for i in range(self.size):
			normm += (self.v[i]**2)
		return math.sqrt(normm)


print(MyVector(4, fill=7).norm())
print(MyVector(4, is_col=False, fill=7).norm())


