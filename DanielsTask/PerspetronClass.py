import numpy as np
import matplotlib.pyplot as plt
import h5py
from unit10 import c1w2_utils as u10

import os
from PIL import Image
import time


class MyPerceptron(object):
    def __init__ (self, X , Y):
        self.X = X
        self.Y = Y
        self.dim = X.shape[0]
        self.m = X.shape[1]
        self.W, self.b = self.initialize_with_zeros(self.dim)
        self.dW, self.db = self.initialize_with_zeros(self.dim)
    
    def sigmoid(self,Z):
        return 1/(1+np.exp(-Z))

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim,1), dtype = float)
        b = 0.0
        return w, b

    def forward_propagation(self):
        Z = np.dot(self.W.T, self.X)+self.b
        A = self.sigmoid(Z) 
        J= (-1/self.m)*np.sum(self.Y * np.log(A) + (1-self.Y) * np.log(1-A))
        J = np.squeeze(J)
        return A, J

    def backward_propagation(self, A):
        dZ = (1/self.m)*(A-self.Y)
        dW = np.dot(self.X, dZ.T)
        db = np.sum(dZ)
        return dW, db

    def train(self, num_iterations, learning_rate):
        for i in range(num_iterations):
            A, cost = self.forward_propagation()
            self.dW, self.db = self.backward_propagation(A)
            self.W -= learning_rate*self.dW
            self.b -= learning_rate*self.db
            #if (i % 100 == 0):
              #print ("Cost after iteration {} is {}".format( i, cost))
        return self.W , self.b

    def predict(self, X, w, b):
        Z = np.dot(w.T,X)+b
        return (np.where(self.sigmoid(Z)>0.5, 1., 0.))




def ahmed(data1):
    ## Loading the data (cat/non-cat)
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = data1

    # setting parameters for the size of the sampled data
    train_set_y = train_set_y.reshape(-1)
    test_set_y = test_set_y.reshape(-1)

    m_train = train_set_y.shape[0]
    m_test = test_set_y.shape[0]
    num_px = train_set_x_orig.shape[1]

    # flatten the pictures to one dimentionsl array of values, keeping a seperated array for each picture
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    train_set_y = train_set_y.reshape(train_set_y.shape[0],-1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    test_set_y = test_set_y.reshape(test_set_y.shape[0],-1).T

    # normalize the values to be between 0 and 1
    train_set_x = train_set_x_flatten/255.0
    test_set_x = test_set_x_flatten/255.0

    return train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes

def createPerceptron(data1):
    train_set_x, train_set_y, test_set_x, test_set_y, num_px, classes = ahmed(data1)

    # create perceptron
    Myperceptron = MyPerceptron(train_set_x, train_set_y)
    
    # train my perceptron
    W, b = Myperceptron.train(num_iterations=4000, learning_rate=0.005)

    # Predict according to the trained perceptron
    Y_prediction_test = Myperceptron.predict(test_set_x, W, b )
    Y_prediction_train = Myperceptron.predict(train_set_x, W, b )

    # Print the accuracy of identifying in the train and in the test
    print ("Percent train = ", np.sum(train_set_y==Y_prediction_train)/train_set_y.shape[1])
    print ("Percent test  = ", np.sum(test_set_y ==Y_prediction_test) /test_set_y.shape[1])

    return Myperceptron, W, b, num_px, classes

def IsCat(data1, fname):
    Myperceptron, W, b, num_px, classes = createPerceptron(data1)
    img = Image.open(fname)
    img = img.resize((num_px, num_px), Image.ANTIALIAS)
    plt.imshow(img)
    my_image = np.array(img).reshape(1, -1).T
    my_predicted_image = Myperceptron.predict(my_image, W, b)
    print("")
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()

IsCat(u10.load_datasetC1W2(), r'C:\Users\danielgonen\Desktop\cat3.jpg')

