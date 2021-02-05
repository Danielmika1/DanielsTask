import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10


MAX_D = 10000000000


def Poli(x, W, low, high):
    y = 0
    for p in range(low,high+1):
        y += x**p*W[p-low]
    return y



def calc_J(X, Y, W, high, low):
    m,n = len(Y),high-low+1
    J = 0
    dW = []
    for j in range(high-low+1):
        dW.append(0)

    for i in range(high-low+1):
        y_hat = Poli(X[i], W, low, high)
        diff = y_hat - Y[i]
        J += (diff**2) / m
        for i in range(n):
            if (j+low == 0):
                dW[j] += (2*diff)/m
            else:
                dW[j] += (2*diff/m)*(j+low)*(X[i]**(j+low-1))
    return J, dW





def train_n_adaptive(X, Y, alpha, num_iterations, calc_J, high, low):
    m,n = len(Y),high-low+1
    costs, W, alpha_W = [],[],[]
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    
    for i in range(num_iterations):
        cost,dW = calc_J(X,Y,W,high,low)
        dW, cost = boundary(dW,cost)

        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j]*alpha_W[j] > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        if ((i%10000)==0):
            print('Iteration :' + str(i) + "  cost= " + str(cost))
            costs.append(cost)
    return costs, W


def boundary1(d):
    return  MAX_D if d > MAX_D else -MAX_D if d < -MAX_D else d

def boundary(dW, db):
    for i in range(len(dW)):
        dW[i] = boundary1(dW[i])
    return dW, boundary1(db)


X = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
Y = [1.2,33693.8,28268.28,13323.43,-3.4,-9837.28,-17444.1,-16850.2,-13581.9,12,20232.34,40051.71,74023.28,96583.9,123678.4,152887.4,141921.4,137721.4,99155.27,17.8,-135173]

p_l , p_h = 0,10
# Power low-high

costs, W = train_n_adaptive(X, Y, 0.00001, 60000, calc_J, p_h, p_l)

plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 1000)')
plt.show()




