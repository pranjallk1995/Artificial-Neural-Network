#Self Implemented Logistic Regression using a single neuron.

#importing libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

#monitoring execution time.
start_time = time.time()

#importing data.
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]]
Y = dataset.iloc[:, 4]

#normalizing data.
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X = X.astype('float')
X = pd.DataFrame(X_sc.fit_transform(X))

#visualizing data.
m = len(Y)
Y_ = np.zeros((m, 1))
fig0, ax0 = plt.subplots()
for i in range(m):
    if Y[i] == 0:
        Y_[i] = 1
ax0.scatter(X[0], X[1], s = 10*Y, c = 'blue', label = 'Has purchased', marker = 'o')
ax0.scatter(X[0], X[1], s = 10*Y_, c = 'red', label = 'Did not purchase', marker = 'x')
ax0.set_title('Data Visualiztion')
ax0.set_xlabel('Age')
ax0.set_ylabel('Estimated Salary')
ax0.legend()

#splitting data.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.25)

#converting to numpy arrays.
X = np.array(X_train)
Y = np.array(Y_train)
m = len(Y_train)
Y = Y.reshape(m, 1)        #making size (400,) to (400, 1)
X_test = np.array(X_test)
Y_test = (np.array(Y_test)).reshape(len(Y_test), 1)

#implementing logistic regression as a single neuron neural network.

#initializing parameters.
W = np.random.rand(2, 1)
b = np.random.rand(1, 1)

#Gradient descent.
iterations = 200
learning_rate = 0.9
parameter_values = np.zeros((3, iterations))
error_values = np.zeros((1, iterations))
iteration_values = np.zeros((1, iterations))

for iters in range(iterations):
    #popagation function of the neuron.
    P = np.dot(X, W) + b
    
    #activation function of the neuron.
    A = 1/(1 + np.exp(-P))
    
    #calculating error.
    E = -(1/m)*(np.dot(Y.T, np.log(A)) + np.dot((1 - Y).T, np.log(1 - A)))
    
    #storing intermediate error values to make graph.
    error_values[0][iters] = E
    
    #calculation of gradient.
    delta_Jw = (1/m)*np.dot((A - Y).T, X)
    delta_Jb = (1/m)*(np.sum(A - Y))
    
    #gradient desccent.
    W = W - learning_rate*(delta_Jw.T)
    b = b - learning_rate*(delta_Jb)
    
    #storing intermediate parameter values to make graph.
    parameter_values[0][iters] = W[0]
    parameter_values[1][iters] = W[1]
    parameter_values[2][iters] = b

    #storing intermediate iteration values to make graph.    
    iteration_values[0][iters] = iters

#plotting the convergence of parameters.
fig1, ax1 = plt.subplots()   
ax1.plot(iteration_values[0], parameter_values[0], color = 'red', label = 'First parameter')
ax1.plot(iteration_values[0], parameter_values[1], color = 'blue', label = 'Second parameter')
ax1.plot(iteration_values[0], parameter_values[2], color = 'green', label = 'Intercept')
ax1.set_title('Parameters values vs Iterations')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Parameter values')
ax1.legend()

#plotting the error values.
fig2, ax2 = plt.subplots() 
ax2.plot(iteration_values[0], error_values[0], color = 'red')
ax2.set_title('Error vaules vs Iterations')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Error values')

#predicting test values.
P = np.dot(X_test, W) + b
Predictions = 1/(1 + np.exp(-P))

for i in range(len(Predictions)):
    if Predictions[i] >= 0.5:
        Predictions[i] = 1
    else:
        Predictions[i] = 0
        
#making confusion matrix.
cm = np.zeros((2, 2))
for i in range(len(Y_test)):
    if Y_test[i] == 1 and Predictions[i] == 1:
        cm[0][0] = cm[0][0] + 1
    elif Y_test[i] == 1 and Predictions[i] == 0:
        cm[0][1] = cm[0][1] + 1
    elif Y_test[i] == 0 and Predictions[i] == 1:
        cm[1][0] = cm[1][0] + 1
    else:
        cm[1][1] = cm[1][1] + 1
        
"""confusion matrix using library:  [[65  3]
                                     [ 8 24]]"""
        
#calculating accuracy.
diagonal_elements = np.diagonal(cm)
num = np.sum(diagonal_elements)
dem = np.sum(cm)        
accuracy = num/dem

#visualization of the hypothesis function.
m = len(Y_test)
Y_ = np.zeros((m, 1))
fig3, ax3 = plt.subplots()
for i in range(m):
    if Y_test[i] == 0:
        Y_[i] = 1
ax3.scatter(X_test[:, 0], X_test[:, 1], s = 10*Y_test, c = 'blue', label = 'Has purchased', marker = 'o')
ax3.scatter(X_test[:, 0], X_test[:, 1], s = 10*Y_, c = 'red', label = 'Did not purchase', marker = 'x')
x = np.arange(-1, 1.5, 0.1)
x = x.reshape(len(x), 1)
y = -W[0]/W[1]*x - b/W[1]
ax3.plot(x, y, color = 'orange')
ax3.set_title('Test data and the hypothesis function')
ax3.set_xlabel('Age')
ax3.set_ylabel('Estimated Salary')
ax3.legend()

#storing execution time.
execution_time = time.time() - start_time

#displaying the graphs.
plt.show()
