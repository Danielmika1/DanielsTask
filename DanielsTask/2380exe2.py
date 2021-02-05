import matplotlib.pyplot as plt
import numpy as np
import random


def Li(a,b,c,xi):
    return a*(xi**2)+b*xi+c

def calc_J(X, Y, a, b, c):
    m = len(Y)
    sumJ = 0
    sumDa = 0
    sumDb = 0
    sumDc = 0
    y_hat_i = 0

    for i in range(m):
        y_hat_i += Li(a,b,c,X[i])
        diff = (float)(y_hat_i - Y[i])
        sumJ += (diff**2)
        sumDa += 2 * diff * (X[i]**2)
        sumDb += 2 * diff * X[i]
        sumDc += 2 * diff
    return sumJ/m, sumDa/m, sumDb/m, sumDc/m


def train_adaptive(X, Y, alpha, epocs, MyFunc):

    a_min = random.randrange(-10,10)
    b_min = random.randrange(-10,10)
    c_min = random.randrange(-10,10)

    alpha_a = alpha
    alpha_b = alpha 
    alpha_c = alpha

    for i in range(epocs):
        Fx, Fda, Fdb, Fdc = MyFunc(X, Y, a_min, b_min, c_min)

        if (Fda * alpha_a < 0):
            alpha_a *= 1.1    #stay same direction
        else:
            alpha_a *= -0.5    #change direction
        a_min += np.abs(Fda) * alpha_a

        if (Fdb * alpha_b < 0):
            alpha_b *= 1.1    #stay same direction
        else:
            alpha_b *= -0.5    #change direction
        b_min += np.abs(Fdb) * alpha_b

        if (Fdc * alpha_c < 0):
            alpha_c *= 1.1    #stay same direction
        else:
            alpha_c *= -0.5    #change direction
        c_min += np.abs(Fdc) * alpha_c

    return Fx, a_min, b_min, c_min


X = [-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
Y = [230.588,160.9912,150.4624,124.9425,127.4042,95.59201,69.67605,40.69738,28.14561,14.42037,7,0.582744,-1.27835,-15.755,-24.692,-23.796,12.21919,9.337909,19.05403,23.83852,9.313449,66.47649,10.60984,77.97216,149.7796,173.2468]
Fx, a_min, b_min, c_min = train_adaptive(X, Y, 0.001, 1000, calc_J)
print('Fx='+str(Fx)+', a_min='+str(a_min)+', b_min='+str(b_min)+', c_min='+str(c_min))

plt.plot([a_min, b_min, c_min])
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()

#X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
#Y = [1.2,33693.8,28268.28,13323.43,-3.4,-9837.28,-17444.1,-16850.2,-13581.9,12,20232.34,40051.71,74023.28,]