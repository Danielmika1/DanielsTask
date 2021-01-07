import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(5) 

def f_function (x):
    Fx = 1.1*x**2-7*x+6
    Fdx = 2.2*x-7
    return Fx, Fdx

def g_function (a):
    Gx = (a**2-9)/(4**a)
    Gdx = ((2**(2*a+1)*a-np.log(2)*2**(2*a+1)*(a**2-9)))/(16**a)
    return Gx, Gdx

def train_min(alpha, epocs, MyFunc):

    # gues the minimum
    xmin = random.randrange(-10,10)

    for i in range (epocs):
        Fx, Fdx = MyFunc(xmin)

        if ((Fdx > 0 and alpha < 0) or (Fdx < 0 and alpha > 0)):
            alpha = alpha * -0.5

        else:
            alpha = alpha * 1.1


        if (Fdx>0):
            xmin -= Fdx*alpha
        else:
            xmin += Fdx*alpha
    Fx, Fdx = MyFunc(xmin)

    return xmin, Fx, Fdx


def train_max(alpha, epocs, MyFunc):
    
    # gues the maxsimum
    xmax = 0
    for i in range (epocs):
        Fx, Fdx = MyFunc(xmax)

        if ((Fdx > 0 and alpha < 0) or (Fdx < 0 and alpha > 0)):
            alpha = alpha * -0.5

        else:
            alpha = alpha * 1.1

        if (Fdx>0):
            xmax += Fdx*alpha
        else:
            xmax -= Fdx*alpha

    Fx, Fdx = MyFunc(xmax)
    return xmax, Fx, Fdx



print("MIN INFO: ", train_min(0.001, 1000, f_function)) 
print("MIN INFO: ", train_min(0.001, 1000, g_function), "MAX INFO: ", train_max(0.001, 1000, g_function))
