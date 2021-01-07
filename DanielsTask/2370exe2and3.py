import matplotlib.pyplot as plt
import numpy as np
import random


def Li(a,b,xi,yi):
    return (a*xi+b-yi)**2

def calc_J(X,Y,a,b):
    m = len(Y)
    sum = 0
    for i in range(m):
        sum += Li(a,b,X[i],Y[i])
    return sum/m

def dJda(X,Y,a,b):
    m = len(Y)
    sum = 0
    for i in range(m):
        sum += 2*X[i]*(a*X[i]+b-Y[i])
    return sum/m

def dJdb(X,Y,a,b):
    m = len(Y)
    sum = 0
    for i in range(m):
        sum += 2*(a*X[i]+b-Y[i])
    return sum/m

X = [10,-3,4]
Y = [12,3,-4]
W = [2,3]

J = calc_J(X,Y,W[0],W[1])
da = dJda(X,Y,W[0],W[1])
db = dJdb(X,Y,W[0],W[1])

print('cost = ' + str(J) + ', da = ' + str(da) + ', db = ' + str(db))