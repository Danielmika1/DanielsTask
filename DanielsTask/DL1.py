import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
from unit10 import c1w2_utils as u10

import os
from PIL import Image
import time
from DL1 import *

class DLLayer(object):
    def __init__ (self, name , num_units, input_shape, activation="relu", W_initialization="random", learning_rate=1.2, optimization=None):
        self.name = name
        self._num_units = num_units
        self._input_shape = input_shape
        self._activation = activation
        self._optimization = optimization
        self.W_initialization = W_initialization
        self.random_scale = 0.01
        self.W, self.b = self.init_weights(W_initialization)
        self.alpha = learning_rate

        self.activation_trim = 1e-10
        if (self._activation == "leaky_relu"):
            self.leaky_relu_d = 0.01

        def _get_W_shape(self):
            return (self._num_units, *(self._input_shape))

        if (self._optimization == 'adaptive'):
            self._adaptive_alpha_b = np.full((self._num_units, 1), self.alpha)
            self._adaptive_alpha_W = np.full(self._get_W_shape(), self.alpha)
            self.adaptive_cont = 1.1
            self.adaptive_switch = 0.5

        if (activation == "sigmoid"):
            self.activation_forward = self._sigmoid
            self.activation_backward = self._sigmoid_backward
        if (activation == "trim_sigmoid"):
           self.activation_forward = self._trim_sigmoid
           self.activation_backward = self._sigmoid_backward
        if (activation == "trim_tanh"):
            self.activation_forward = self._trim_tanh
            self.activation_backward = self.trim_tanh_backward
        if (activation == "tanh"):
            self.activation_forward = self._tanh
            self.activation_backward = self._tanh_backward
        if (activation == "relu"):
            self.activation_forward = self._relu
            self.activation_backward = self._relu_backward
        if (activation == "leaky_relu"):
            self.activation_forward = self._leaky_relu
            self.activation_backward = self._leaky_relu_backward

    ## activations and backwards, exe 1.2 and exe 1.4 ##
    
    def _sigmoid(self, Z):
        A = 1/(1+np.exp(-Z))
        return A

    def _sigmoid_backward(self, dA):
        A = self._sigmoid(self._Z)
        dZ = dA * A * (1-A)
        return dZ

    def _tanh(self, Z):
        A = np.tanh(Z)
        return A

    def _tanh_backward(self, dA):
        A = self._tanh(self._Z)
        dZ = dA * (1-A**2)
        return dZ

    def _relu(self, Z):
        A = np.maximum(0,Z)
        return A

    def _relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, 0, dA)
        return dZ

    def _leaky_relu(self, Z):
        A = np.where(Z > 0, Z, self.leaky_relu_d * Z)
        return A

    def _leaky_relu_backward(self, dA):
        dZ = np.where(self._Z <= 0, self.leaky_relu_d * dA, dA)
        return dZ
    
    def _trim_sigmoid(self,Z):
        with np.errstate(over='raise', divide='raise'):
            try:
               A = 1/(1+np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100,Z)
                A = A = 1/(1+np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def _trim_sigmoid_backward(self, dA):
        A = self._trim_sigmoid(self._Z)
        return dZ

    def _trim_tanh(self,Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1+TRIM,TRIM,A)
            A = np.where(A > 1-TRIM,1-TRIM, A)
        return A

    def trim_tanh_backward(self, dA):
        A = self.trim_tanh(self._Z)
        dZ = dA * (1-A**2)
        return dZ

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self._num_units) + "\n"
        s += "\tactivation: " + self._activation + "\n"
        if self._activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
            s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d)+"\n"
        s += "\tinput_shape: " + str(self._input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.alpha) + "\n"
        #optimization
        if self._optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            s += "\t\t\tcont: " + str(self.adaptive_cont)+"\n"
            s += "\t\t\tswitch: " + str(self.adaptive_switch)+"\n"
        # parameters
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape)+"\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s

    def init_weights(self, W_initialization):
        self.b = np.zeros((self._num_units,1), dtype=float)

        if (W_initialization == "zeros"): 
            self.W = np.zeros(self._get_W_shape(), self.alpha)

        elif(W_initialization == "random"):
            self.W = np.random.randn(self._num_units, *(self._input_shape)) * self.random_scale

        return self.W, self.b

    def forward_propagation(self, A_prev, is_predict):
        self._A_prev = np.array(A_prev, copy=True)
        self._Z = self.W @ A_prev + self.b
        A = self.activation_forward(self._Z)
        return A

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        m = self._A_prev.shape[1]
        self.db = (1/m)*(np.sum(dZ , axis=1, keepdims=True) )
        self.dW = (1/m)*(dZ @ (self._A_prev ).T)
        dA_Prev = (self.W).T @ dZ

        return dA_Prev

    def update_parameters(self):
        self.W = self.dW
        self.b = self.db

        if (self._optimization == "None"):
            self.W, self.b = self.dW * self.alpha, self.db * self.alpha
        if (self._optimization == "adaptive"):
            self._adaptive_alpha_W *= np.where(_adaptive_alpha_W * self.dW > 0, self.adaptive_cont, -adaptive_switch)
            self._adaptive_alpha_b *= np.where(_adaptive_alpha_b * self.db > 0, self.adaptive_cont, -adaptive_switch)
            self.W, self.b = self.W * self._adaptive_alpha_W, self.b * self._adaptive_alpha_b

        return self.W, self.b



###--------------------------------------###
###-------------- cheacking -------------###
###--------------------------------------###


# 1.1 

np.random.seed(1)

l = [None]
l.append(DLLayer("Hidden 1", 6, (4000,)))
print(l[1])

l.append(DLLayer("Hidden 2", 12,(6,),"leaky_relu", "random", 0.5,"adaptive"))
l[2].adaptive_cont = 1.2
print(l[2])

l.append(DLLayer("Neurons 3",16, (12,),"tanh"))
print(l[3])

l.append(DLLayer("Neurons 4",3, (16,),"sigmoid","random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
print(l[4])

# 1.2
Z = np.array([[1,-2,3,-4],
[-10,20,30,-40]])
l[2].leaky_relu_d = 0.1
for i in range(1, len(l)):
    print(l[i].activation_forward(Z))

# 1.3
np.random.seed(2)
m = 3
X = np.random.randn(4000,m)
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    print('layer',i," A", str(Al.shape), ":\n", Al)

# 1.4
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    dZ = l[i].activation_backward(Al)
    print('layer',i," dZ", str(dZ.shape), ":\n", dZ)

# 1.5
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)

np.random.seed(3)
fig, axes = plt.subplots(1, 4, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
dAl = np.random.randn(Al.shape[0],m) * np.random.random_integers(-100, 100, Al.shape)
for i in reversed(range(1,len(l))):
    axes[i-1].hist(dAl.reshape(-1), align='left')
    axes[i-1].set_title('dAl['+str(i)+']')
    dAl = l[i].backward_propagation(dAl)
plt.show()

# 1.6 
np.random.seed(4)
random.seed(4)
l1 = DLLayer("Hidden1", 3, (4,),"trim_sigmoid", "zeros", 0.2, "adaptive")
l2 = DLLayer("Hidden2", 2, (3,),"relu", "random", 1.5)

print("before update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

l1.dW = np.random.randn(3,4) * random.randrange(-100,100)
l1.db = np.random.randn(3,1) * random.randrange(-100,100)
l2.dW = np.random.randn(2,3) * random.randrange(-100,100)
l2.db = np.random.randn(2,1) * random.randrange(-100,100)
l1.update_parameters()
l2.update_parameters()
print("after update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))