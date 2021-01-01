
import matplotlib.pyplot as plt
import numpy as np

import random
random.seed(1)

def f_function (x):
    Fx = x**3 - 107*(x**2) - 9*x + 3
    Fdx = 3*(x**2) - 214*x - 9
    return Fx, Fdx


def train_min(alpha, epocs, MyFunc):
    xmin = 0

    for i in range (epocs):
        Fx, Fdx = MyFunc(xmin)
        if (Fdx>0):
            xmin -= Fdx*alpha
        else:
            xmin += Fdx*alpha
    Fx, Fdx = MyFunc(xmin)
    return xmin, Fx, Fdx


def train_max(alpha, epocs, MyFunc):
    xmax = 0

    for i in range (epocs):
        Fx, Fdx = MyFunc(xmax)
        if (Fdx>0):
            xmax += Fdx*alpha
        else:
            xmax -= Fdx*alpha
    Fx, Fdx = MyFunc(xmax)
    return xmax, Fx, Fdx

print("MIN INFO: ", train_min(0.001, 100, f_function), "\nMAX INFO: ", train_max(0.001, 100, f_function))