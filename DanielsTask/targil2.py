import matplotlib.pyplot as plt
import numpy as np
import unit10.b_utils as u10
import random

random.seed(1)

X, Y = u10. load_dataB1W3Ex1()
plt.plot(X, Y, 'b.')
##plt.show()

line = (min(X), max(X))

aLst = []
bLst = []

for i in range(10):
    aLst.append(random.randrange(-20,20))
    bLst.append(random.randrange(-20,20))

for i in range(len(aLst)):
    plt.plot(line,[line[0]*aLst[i],line[1]*aLst[i]+bLst[i]])

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



#plt.plot(line,[line[0]*a+b, line[1]*a+b])
plt.show()








