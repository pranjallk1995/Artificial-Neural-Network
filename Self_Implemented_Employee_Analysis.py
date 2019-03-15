# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:23:07 2019

@author: Pranjall
"""

import numpy as np
import pandas as pd
import time


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def ReLU(x):
    return np.maximum(x, 0, x)

def sigmoid_derivative(P):
    return P * (1 - P)

def ReLU_derivative(P):
    P[P <= 0] = 0
    P[P > 0] = 1
    return P

def classify(P):
    P[P < 0.5] = 0
    P[P >= 0.5] = 1
    return P


class NeuralNetwork:
    
    def __init__(self, x, y):
        
        self.input = x
        self.y = y
        self.nodes_in_first_layer = 8
        self.nodes_in_second_layer = 6
        self.nodes_in_third_layer = 6
        self.nodes_in_output_layer = 1
        self.output = np.zeros(y.shape[0])
        
        upper_limit = 0.5
        lower_limit = -0.5
        
        self.weights1 = np.random.uniform(upper_limit, lower_limit, (self.input.shape[1], self.nodes_in_first_layer))
        self.weights2 = np.random.uniform(upper_limit, lower_limit, (self.nodes_in_first_layer, self.nodes_in_second_layer))
        self.weights3 = np.random.uniform(upper_limit, lower_limit, (self.nodes_in_second_layer, self.nodes_in_third_layer))
        self.weights4 = np.random.uniform(upper_limit, lower_limit, (self.nodes_in_third_layer, self.nodes_in_output_layer))
        
        self.bias1 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        self.bias2 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        self.bias3 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        self.bias4 = np.random.uniform(upper_limit, lower_limit, (1, 1))
        
    def forwardprop(self):
        
        self.layer1 = ReLU(np.dot(self.input, self.weights1) + self.bias1)
        self.layer2 = ReLU(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = ReLU(np.dot(self.layer2, self.weights3) + self.bias3)
        self.layer4 = sigmoid(np.dot(self.layer3, self.weights4) + self.bias4)
        
        return self.layer4
    
    def error(self):
        return -(1 / self.output.shape[0]) * ((self.y * np.log(self.output)) + ((1 - self.y) * np.log(1 - self.output)))
    
    def backprop(self):
        
        learning_rate = 0.3
        
        d_Propogation4 = self.output - self.y
        d_weights4 = (1 / self.output.shape[0]) * (np.dot(d_Propogation4.T, self.layer3))
        d_bias4 = (1 / self.output.shape[0]) * (np.sum(d_Propogation4))
        
        d_Propogation3 = np.dot(d_Propogation4, self.weights4.T) * ReLU_derivative(self.layer3)
        d_weights3 = (1 / self.output.shape[0]) * (np.dot(d_Propogation3.T, self.layer2))
        d_bias3 = (1 / self.output.shape[0]) * (np.sum(d_Propogation3))
        
        d_Propogation2 = np.dot(d_Propogation3, self.weights3.T) * ReLU_derivative(self.layer2)
        d_weights2 = (1 / self.output.shape[0]) * (np.dot(d_Propogation2.T, self.layer1))
        d_bias2 = (1 / self.output.shape[0]) * (np.sum(d_Propogation2))
        
        d_Propogation1 = np.dot(d_Propogation2, self.weights2.T) * ReLU_derivative(self.layer1)
        d_weights1 = (1 / self.output.shape[0]) * (np.dot(d_Propogation1.T, self.input))
        d_bias1 = (1 / self.output.shape[0]) * (np.sum(d_Propogation1))
        
        self.weights4 = self.weights4 - learning_rate * d_weights4.T
        self.bias4 = self.bias4 - learning_rate * d_bias4
        
        self.weights3 = self.weights3 - learning_rate * d_weights3.T
        self.bias3 = self.bias3 - learning_rate * d_bias3
        
        self.weights2 = self.weights2 - learning_rate * d_weights2.T
        self.bias2 = self.bias2 - learning_rate * d_bias2
        
        self.weights1 = self.weights1 - learning_rate * d_weights1.T
        self.bias1 = self.bias1 - learning_rate * d_bias1
        
    def simulate(self):
        
        self.output = self.forwardprop()
        self.backprop()
        print(self.error())
        
        return self.output
    
    def run(self, test):
        
        self.layer1 = ReLU(np.dot(test, self.weights1) + self.bias1)
        self.layer2 = ReLU(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = ReLU(np.dot(self.layer2, self.weights3) + self.bias3)
        self.layer4 = sigmoid(np.dot(self.layer3, self.weights4) + self.bias4)
        
        return self.layer4
    



if __name__ == '__main__':
    
    #importing the data.
    dataset = pd.read_csv("Churn_Modelling.csv")
    X = dataset.iloc[:, 3: 13]
    Y = dataset.iloc[:, 13]
    
    #encoding categorical data.
    X = pd.get_dummies(X, columns = ['Geography'], prefix = ['Country'])            # This method is called One-Hot encoding.
    X = X.iloc[:, :-1]              #avoiding dummy veriable trap.
    
    #enooding binary data.
    from sklearn.preprocessing import LabelEncoder
    classifier_encoder = LabelEncoder()
    X["Gender"] = classifier_encoder.fit_transform(X["Gender"])

    #feature scaling.
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = X.astype('float')
    X = sc.fit_transform(X)
    
    #splitting data.
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    
    #converting to numpy arrays.
    X = np.array(X_train)
    Y = np.array(Y_train)
    m = len(Y)
    Y = Y.reshape(m, 1)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    m = len(Y_test)
    Y_test = Y_test.reshape(m, 1)
    
    #simulating the Neural Network.
    NN = NeuralNetwork(X, Y)
    iterations = 5000
    
    #monitoring execution time.
    start_time = time.time()

    for i in range(0, iterations):
        print('Iteration: ' + str(i) + ', Error: ', end = ' ')
        output = NN.simulate()
        
    #storing execution time.
    execution_time = time.time() - start_time
    
    output = classify(output)
    
    #simulating the Neural Network on test data.
    result = NN.run(X_test)
    classify(result)
    
    count = 0
    for i in range(len(Y_test)):
        if result[i] != Y_test[i]:
            count = count + 1
            
    accuracy = (len(Y_test) - count)/len(Y_test)
