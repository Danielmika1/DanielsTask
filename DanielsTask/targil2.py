import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random
from mpl_toolkits import mplot3d

random.seed(1)
X, Y = u10. load_dataB1W3Ex2()

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
