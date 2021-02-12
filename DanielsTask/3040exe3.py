import numpy as np
import time
import matplotlib.pyplot as plt
import unit10.b_utils as u10

def calc_J_np_v2 (X, Y, W, b):
    m,n = len(Y),len(X)
    dW = np.zeros((n,1))
    J,db = 0,0
    for i in range(m):
        Y_hat = np.dot(W.T,X) + b
        diff = Y_hat-Y
        J = np.sum(diff**2)/m
        dW = np.sum(2*X*diff, axis=1,keepdims=True)/m
        db = np.sum(2*diff)/m
    return J, dW, db


X, Y = u10.load_dataB1W4_trainN()
np.random.seed(1)
J, dW, db = calc_J_np_v2(X,Y,np.random.randn(len(X),1),3)
print(J)
print(dW.shape)
print(db)
