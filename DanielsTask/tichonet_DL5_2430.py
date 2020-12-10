import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
from mpl_toolkits import mplot3d

random.seed(1)
X, Y = u10. load_dataB1W3Ex1()

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
    return a*(x1*2) + b*x1*x2 + c*(x2**2) + d

indexMin = -1
min = 0

for l in range(len(aLst)):
    sum = 0;
    for i in range(len(X)):
        sum += (aLst[l]*X[i]+bLst[l]-Y[i])**2
    if(indexMin==-1):
        min = sum/len(X)
        indexMin = l
    elif (min>sum/len(X)):
          min = sum/len(X)
          indexMin = 1


fig = plt.figure()
ax = plt.axes(projection="3d")
X1, X2 = np.meshgrid(np.linspace(-15, 15, 30), np.linspace(-15, 15, 30))
Ywire = Guess(X1, X2)
ax.plot_wireframe(X1, X2, Ywire, color='orange')
ax.scatter3D(X[0], X[1], Y);
plt.show()
