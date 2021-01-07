import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(5) 

def f_function (x):
    Fx = (2*x + 1) * 2
    Fdx = 8*x + 4
    return Fx, Fdx

def g_function (x, a=3, b=5):
    Gx = (2*x - a) * 2 + (3*x - b) * 2
    Gdx = 4*(2*x - a) + 6 * (3*x - b)
    return Gx, Gdx

def k_function (a, m=10):
    sum_Kx = 0
    sum_Kdx = 0

    for i in range(1, m + 1):
        sum_Kx += (a - i)**2
        sum_Kdx += 2 * (a - i)

    Kx = sum_Kx / m
    Kdx = sum_Kdx / m 

    return Kx, Kdx

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

    return xmin


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
print("MIN INFO: ", train_min(0.001, 1000, g_function))
print("MIN INFO: ", train_min(0.001, 1000, k_function))

