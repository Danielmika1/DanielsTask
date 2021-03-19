import matplotlib.pyplot as plt
import time
import random
import numpy as np
import unit10.c1w2_utils as u10
from PIL import Image

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

train_set_y = train_set_y.reshape(-1)
test_set_y = test_set_y.reshape(-1)

m_train = train_set_y.shape[0]
m_test = test_set_y.shape[0]
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_set_y = train_set_y.reshape(train_set_y.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_y = test_set_y.reshape(test_set_y.shape[0],-1).T

train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim): # fill with 0
    W = np.zeros((dim,1))
    b = 0
    return W,b

def forward_propagation(X,Y,W,b):
    m = X.shape[1]
    A = sigmoid(W.T@X+b)
    J = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return A, J

def backward_propagation(X, Y, A):
    m = X.shape[1]
    dZ = (A-Y)/m
    dW = np.dot(X, dZ.T)
    db = np.sum(dZ)
    return dW, db


def train(X, Y, num_iterations, learning_rate):
    W, b = initialize_with_zeros(len(X))
    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)
        W -= learning_rate * dw
        b -= learning_rate * db
    return W, b

def predict(X, w, b):
        Z = np.dot(w.T,X)+b
        return (np.where(sigmoid(Z)>0.5, 1., 0.))


def IsCat(fname, W, b):
    img = Image.open(fname)
    img = img.resize((num_px, num_px), Image.ANTIALIAS)
    plt.imshow(img)
    my_image = np.array(img).reshape(1, -1).T
    my_predicted_image = predict(my_image, W, b)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()


W, b = train(train_set_x, train_set_y, num_iterations=5000, learning_rate=0.0005)
fname1 = r'C:\Users\danielgonen\Desktop\cat.jpg'  # <=== change image full path
fname2 = r'C:\Users\danielgonen\Desktop\cat2.jpg'  # <=== change image full path
fname3 = r'C:\Users\danielgonen\Desktop\cat3.jpg'  # <=== change image full path
fname4 = r'C:\Users\danielgonen\Desktop\cat4.jpg'  # <=== change image full path

IsCat(fname1, W, b)
IsCat(fname2, W, b)
IsCat(fname3, W, b)
IsCat(fname4, W, b)

Y_prediction_test = predict(test_set_x, W, b)
Y_prediction_train = predict(train_set_x, W, b)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))