import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


class MyVector():
    def __init__ (self, size, is_col=True, fill=0, init_values=None):
        self.vector = []
        self.size = size
        self.is_col = is_col
        if (init_values != None):
            l = len(init_values)
            for i in range(size):
                self.vector.append(init_values[i % l])
        else:
            for i in range (size):
                self.vector.append(fill)

    def __str__(self):
        s = '['
        lf = "\n" if self.is_col else ""
        for item in self.vector:
            s = s + str(item) + ', ' +lf
        s+= "]"
        return  (s)



print(MyVector(4,fill=7))
print(MyVector(5,is_col=False,init_values=[1,2,3]))