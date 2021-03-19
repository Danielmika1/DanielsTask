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

print(train_set_y.shape)

# Example of a picture
index = 5 # change index to get a different picture
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[index]) + ", it's a '" + 
classes[np.squeeze(train_set_y[index])].decode("utf-8") +  "' picture.")

m_train = train_set_y.shape[0]
m_test = test_set_y.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
train_set_y = train_set_y.reshape(train_set_y.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_y = test_set_y.reshape(test_set_y.shape[0],-1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x = train_set_x_flatten/255.0
test_set_x = test_set_x_flatten/255.0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim): # fill with 0
    W = np.zeros((dim,1))
    b = 0
    return W,b

W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print ("b = " + str(b))

def forward_propagation(X,Y,W,b):
    m = X.shape[1]
    A = sigmoid(W.T@X+b)
    J = -1/m*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    return A, J

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1]) 
A, cost = forward_propagation(X, Y, w, b)
print ("cost = " + str(cost))

def backward_propagation(X, Y, A):
    m = X.shape[1]
    dZ = (A-Y)/m
    dW = np.dot(X, dZ.T)
    db = np.sum(dZ)
    return dW, db

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1]) 
A, cost = forward_propagation(X, Y, w, b)
dw, db = backward_propagation(X, Y, A)
print ("dW = " + str(dw))
print ("db = " + str(db))

def train(X, Y, num_iterations, learning_rate):
    W, b = initialize_with_zeros(len(X))

    for i in range(num_iterations):
        A, cost = forward_propagation(X, Y, W, b)
        dw, db = backward_propagation(X, Y, A)
        W -= learning_rate * dw
        b -= learning_rate * db

    return W, b

X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1]) 
W, b = train(X, Y, num_iterations= 100, learning_rate = 0.009)
print ("W = " + str(W))
print ("b = " + str(b))

def predict(X, w, b):
        Z = np.dot(w.T,X)+b
        return (np.where(sigmoid(Z)>0.5, 1., 0.))

W = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(X, W, b)))

W, b = train(train_set_x, train_set_y, num_iterations=4000, learning_rate=0.005)
Y_prediction_test = predict(test_set_x, W, b)
Y_prediction_train = predict(train_set_x, W, b)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

fname = r'C:\Users\danielgonen\Desktop\cat3.jpg'  # <=== change image full path
img = Image.open(fname)
img = img.resize((num_px, num_px), Image.ANTIALIAS)
plt.imshow(img)
plt.show()
my_image = np.array(img).reshape(1, -1).T
my_predicted_image = predict(my_image, W, b)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

