import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
from mpl_toolkits import mplot3d


X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
Y =  [1.2,33693.8,28268.28,13323.43,-3.4,-9837.28,-17444.1,-16850.2,-13581.9,12,20232.34,40051.71,74023.28,96583.9,123678.4,152887.3,141921.4,137721.4,99155.27,17.8,-135173]
W = [-1.2,31.2,-150,1320,-31200,150000,-120000]



def func_etgar1(x):
    return -1.2(x**5) + 31.2(x**4) - 150(x**3) + 1320(x**2) - 31200*x + (150000 - (120000/x))

for l in range(len(aLst)):
    sum = 0.0
    for i in range(len(Y)):
        sum += (Guess(X[0][i], X[1][i], aLst[l], bLst[l], cLst[l], dLst[l])-Y[i])**2
    if(indexMin == -1):
        min = sum/len(Y)
        indexMin = l
    elif (min>sum/len(Y)):
          min = sum/len(Y)
          indexMin = 1


aLst = []
bLst = []
cLst = []
dLst = []

for i in range(1000):
    aLst.append(random.randrange(-10,10))
    bLst.append(random.randrange(-10,10))
    cLst.append(random.randrange(-10,10))
    dLst.append(random.randrange(-10,10))

def Guess(x1,x2,a,b,c,d):
    return a*(x1**2) + b*x1*x2 + c*(x2**2) + d

indexMin = -1
min = 0

for l in range(len(aLst)):
    sum = 0.0
    for i in range(len(Y)):
        sum += (Guess(X[0][i], X[1][i], aLst[l], bLst[l], cLst[l], dLst[l])-Y[i])**2
    if(indexMin == -1):
        min = sum/len(Y)
        indexMin = l
    elif (min>sum/len(Y)):
          min = sum/len(Y)
          indexMin = 1

#print("a={0}, b={1}, c={2}, d={3}, cost={4}".format(aLst[indexMin], bLst[indexMin], cLst[indexMin], dLst[indexMin], min))


fig = plt.figure()
ax = plt.axes(projection="3d")
X1, X2 = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
Ywire = Guess(X1, X2, aLst[indexMin], bLst[indexMin], cLst[indexMin], dLst[indexMin])
ax.plot_wireframe(X1, X2, Ywire, color='orange')
ax.set_title("a={0}, b={1}, c={2}, d={3}, cost={4}".format(aLst[indexMin], bLst[indexMin], cLst[indexMin], dLst[indexMin], min))
ax.scatter3D(X[0], X[1], Y);
plt.show()
