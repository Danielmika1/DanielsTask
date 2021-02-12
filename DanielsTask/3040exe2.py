import numpy as np
import time
import matplotlib.pyplot as plt
import unit10.b_utils as u10

def calc_J_np_v1 (X, Y, W, b):
    m,n = len(Y),len(X)
    dW = np.zeros((n,1))
    J,db = 0,0
    for i in range(m):
        Xi = X[:,i].reshape(len(W),1)
        y_hat_i = np.dot(W.T,Xi) + b
        diff = (float)(y_hat_i - Y[i])
        J += diff**2/m
        dW += (2*diff/m)*Xi
        db += 2 * diff/m
    return J, dW, db


X, Y = u10.load_dataB1W4_trainN()
np.random.seed(1)
J, dW, db = calc_J_np_v1(X,Y,np.random.randn(len(X),1),3)
print(J)
print(dW.shape)
print(db)
