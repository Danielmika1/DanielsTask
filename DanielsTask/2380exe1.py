import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10

random.seed(5)

""" alpha aka learning rate """

def Li(a,b,xi,yi):
    return (a*xi+b-yi)**2

def calc_J(X, Y, W, b):
    m = len(Y)
    n = len(W)
    J = 0
    dW = []
    for j in range(n):
      dW.append(0)
    db = 0

    for i in range(m):
      y_hat_i = b
      for j in range(n):
        y_hat_i += W[j]*X[j][i]
      diff = (float)(y_hat_i - Y[i])
      J += (diff**2)/m
      for j in range(n):
        dW[j] += (2*diff/m)*X[j][i]
      db += 2 * (diff);
    return J, dW, db/m




def train_n_adaptive(X, Y, alpha, num_iterations, calc_J):
    m,n = len(Y), len(X)
    x = 1
    costs, W, alpha_W,b = [],[],[],0
    for j in range(n):
        W.append(0)
        alpha_W.append(alpha)
    alpha_b = alpha
    
    for i in range(num_iterations):
        cost,dW,db = calc_J(X,Y,W,b)
        if(i == (10000*x)-1):
            print("J="+str(cost))
            x = x+1
        for j in range(n):
            alpha_W[j] *= 1.1 if dW[j]*alpha_W[j] > 0 else -0.5
        alpha_b *= 1.1 if db*alpha_b > 0 else -0.5
        for j in range(n):
            W[j] -= alpha_W[j]
        b -= alpha_b
        if i%(num_iterations//10)==0:
            costs.append(cost)
    return costs, W, b



X, Y = u10.load_dataB1W4_trainN()
costs, W, b = train_n_adaptive(X, Y, 0.0001, 150000, calc_J)
print('w1='+str(W[0])+', w2='+str(W[1])+', w3='+str(W[2])+', w4='+str(W[3])+", b="+str(b))
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 10,000)')
plt.show()
