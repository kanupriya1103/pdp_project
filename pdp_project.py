import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
import pandas as pd

hello=pd.read_csv("creditcard2.csv")
print(hello)
y=hello[['Class']].to_numpy()
y_train=y[:227845][:]
y_train=y_train.T
print(y_train)
y_test=y[227845:][:]
y_test=y_test.T

print(y_test)

x=hello.to_numpy()
x=np.delete(x,0,1)
x=np.delete(x,29,1)

x_train=x[:227845][:]
x_train=x_train.T
print(x_train)
x_test=x[227845:][:]
x_test=x_test.T
print(x_test)



def sigmoid(z):
    s=1/(1+np.exp(-z))
    print(s.shape)
    return s

def initialization(n_x):
    w = np.zeros((n_x, 1))
    b = 0.0
    print("w=")
    print(w.shape)
    return w,b


def propagate(w, b, x, y):
    
   
    Z=np.dot(w.T,x)+b
    print("z=")
    print(Z.shape)
    A=sigmoid(Z)
    
    return A

def gradient_descent(w,b,x,y,n_i,lr):
    
    m=w.shape[1]
    print(x.shape)
    
    for i in range(n_i):
        A=propagate(w,b,x,y)
        print("A=")
        print((A-y).shape)
    
        cost=np.sum(np.sum(((- np.log(A))*y + (-np.log(1-A))*(1-y)))/m)

        dw = (np.dot(x,(A-y).T))/m
        db = (np.sum(A-y))/m
        
        w = w - (lr*dw)
        b = b - (lr*db)
        
        print(w.shape)
        
    return w,b

def prediction(w,b,x):
    m = x.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)
    
    A = sigmoid(w.T.dot(x) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] <= 0.5 :
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction

def model(x_train, y_train, x_test, y_test, n_i , lr ):
    n_i=2000
    lr=0.5
    w,b=initialization(x_train.shape[0])
    
    w,b=gradient_descent(w, b, x_train, y_train, n_i, lr)
    
    print(w.shape)
    
    Y_prediction_test = prediction(w, b, x_test)
    Y_prediction_train = prediction(w, b, x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - y_test)) * 100))
    
model(x_train, y_train, x_test, y_test, n_i = 2000, lr = 0.5)