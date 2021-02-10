import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


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
		w = self.CheckErr(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] + w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))

	def __mul__(self,w):
		w = self.CheckErr(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] * w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))


	def __truediv__(self,w):
		w = self.CheckErr(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] / w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))

	def __sub__(self, w):
		w = self.CheckErr(w)
		ans = []
		for i in range(self.size):
			ans.append(self.v[i] - w.v[i])
		return(MyVector(self.size, self.is_col, fill=0, init_values=ans))


	def CheckErr(self, err):
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


	def __radd__(self, number):
		return self + number




print(MyVector(4,fill=7))
print(MyVector(5,is_col=False,init_values=[1,2,3]))
print(MyVector(4,init_values=[4,1,2,4]) + MyVector(4,fill=7))

print("**********************")

print(MyVector(4,init_values=[4,1,2,4]) - MyVector(4,fill=7))

print("**********************")

print(MyVector(4,init_values=[4,1,2,4]) * MyVector(4,fill=7))

print("**********************")

print(MyVector(4,init_values=[4,1,2,4]) / MyVector(4,fill=7))

print("**********************")

try: 
	print(MyVector(3,fill=1) + MyVector(4,fill=2))
except Exception as err1:
	print("Exception:", err1)
	try: 
		print(MyVector(3,is_col=False,fill=1) + MyVector(3,fill=2))
	except Exception as err2:
		print("Exception:", err2)
		print(MyVector(3,fill=15) + MyVector(3,fill=21))

print("**********************")

v = MyVector(3,init_values=[-0.7,-0.03,1.44])
vT = v.transpose()
print(v,vT)

print("**********************")

x = MyVector(3, fill=15)
x[2] = 7
print(x[2])

print("**********************")

print(len(v))

print("**********************")

v = MyVector(3,False,init_values=[-0.7,-0.03,1.44])
print(v, (15+(v-4)*3)/2)



