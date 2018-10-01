import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import autograd
# from mxnet import ndarray as nd
from numpy import random
import math

# Load data
x = pd.read_csv("X_train.csv", sep=',')
y = pd.read_csv("Y_train.csv", sep=',')

x_train = pd.DataFrame(x)
x_train.loc[5331] = x_train.columns
y_train = pd.DataFrame(y)
y_train.loc[5331] = y_train.columns

"""
Create columns:
age: subject age in years
sex: subject gender, '0'- male, '1'- female
Jitter(%), Jitter(Abs), Jitter: RAP, Jitter: PPQ5, Jitter: DDP: Several measures of variations in fundamental frequency of voice 
Shimmer...: Several measures of variation in amplitude of voice
NHR, HNR: Two measures of ratio of noice to tonal components in the voice 
RPDE: A nonlinear dynamical complexity measure
DFA: Signal fractal scaling exponent
PPE: A nonlinear measure of fundamental frequency variation
"""
x_train.columns = ['age', 'sex', 'Jitter(%)', 'Jitter(abs)','Jitter(RAP)', 'Jitter(PPQ5)','Jitter(DDP)','Shimmer','Shimmer(dB)','Shimmer(APQ3)','Shimmer(APQ5)','Shimmer(APQ11)','Shimmer(DDA)','NHR','HNR','RPDE','DFA','PPE']
x_train.head()
num = y_train.shape[0]

##Derive the stochastic gradient descent 
def init_params():
    beta = np.random.normal(scale = 1, size=(x_train.shape[1],1))
    beta_0 = np.zeros(shape=(1,))
    params = [beta, beta_0]
    return params

%matplotlib inline
# Stochastic gradient descent 
def sgd(data, t_k, beta, label, group_size):
    g_prime = np.dot(data.T, (np.dot(data, beta) - label))/group_size
    beta_update = beta - g_prime * t_k
    return beta_update
        
# construct data iterator 
def get_mini_batches(X, y, batch_size):
    random_idxs = list(range(X.shape[0]))
    random.shuffle(random_idxs)
    mini_batches = [(X.loc[np.array(random_idxs[i:(i+batch_size)])], y.loc[np.array(random_idxs[i:(i+batch_size)])]) for i in range(0, len(y), batch_size)]
    return mini_batches
        
# linear regression
def linear_regression(X, beta, beta_0):
    return np.dot(X, beta) + beta_0

# Loss Function
def square_loss(yhat, y, batch_size, beta):
    return (np.sum((yhat-y)**2))/(2*batch_size) + (1/2)*math.pow(np.linalg.norm(np.matrix(beta,dtype='float'),2),2)


def train(batch_size, t_k, epochs, lmd, n):
    beta_k, beta_0_k = init_params()
    total_loss = []
    # Epoch starts from 1
    #data, label = data_iter(group_size)
    mini_batches = get_mini_batches(x_train, y_train, batch_size)
    for mb in mini_batches:
        data = mb[0]
        label = mb[1]
        for step in range(1, epochs+1):
            output = linear_regression(data, beta_k, beta_0_k)
            loss = square_loss(output, label, batch_size, beta_k)
            beta_k_plus_1 = sgd(data, t_k, beta_k, label, batch_size)
            output = linear_regression(data, beta_k_plus_1, beta_0_k)
            loss_plus_1 = square_loss(output, label, batch_size, beta_k_plus_1)
            if loss[0] >= loss_plus_1[0]:
                total_loss.append(loss-57.0410)
                beta_k = beta_k_plus_1
            else:
                break
    x_axis = np.linspace(0, epochs, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


for i in (10,20,50,100):
    for j in (0.01,0.001,0.0001,0.00001):
        train(batch_size=i, t_k=j, epochs=500, lmd=1, n=5000)

## b. Proximal Gradient Descent
import math
from numpy import linalg as LA

## Proximal function
# a) Derive the proximal operator prox_ht for the non-smooth component
##Derive the stochastic gradient descent 
# construct data iterator 
# def vec_max(value, vector):
#     for i in range(len(vector)):
#         vector[i] = max(value, vector[i])
#     return vector

def init_params():
    beta = np.random.normal(scale = 1, size=(18,1))
    beta_0 = np.zeros(shape=(1,))
    params = [beta, beta_0]
    return params

def h_func(beta,group_size,lmd):
    p = math.sqrt(group_size)
    h_value = lmd*p*LA.norm(np.matrix(beta,dtype='float'),2)
    return h_value
    
def f_func(beta,data,n,y):
    f_value = math.pow(LA.norm(np.dot(data,beta) - y),2)/(2*n)
    return f_value

def obj_func(data,beta,group_size,label,lmd):
    f_value = f_func(beta,data,group_size,label)
    h_value = h_func(beta,group_size,lmd)
    value = f_value + h_value
    return value
    
# def prox_obj(lmd, group_size, beta_update, t_k):
#     p = math.sqrt(group_size)
#     prox_vec = (beta_update / (1/2 * t_k + lmd * p))/ (2*t_k)
#     return prox_vec
    
def gradient_descent(data, t_k, beta, label, group_size):
    g_prime = np.dot(data.T, (np.dot(data, beta) - label))/group_size
    beta_update = beta - g_prime * t_k
    return beta_update
    
def m_func(data,beta_k, beta_update, group_size,label,lmd,t_k):
    part_1 = f_func(beta_k,data,group_size,label)
    part_2 = t_k * np.dot(gradient_descent(data, t_k, beta_k, label, group_size).T, (beta_k - beta_update)/t_k)
    #part_3 = t_k * math.pow(LA.norm(np.matrix((beta_k - prox_obj(lmd, group_size, beta_update, t_k))/t_k,dtype='float'),2),2)/2
    m_value = part_1 - part_2
    return m_value

    def train(group_size, t_k, steps, lmd, n):
    beta_k, beta_0_k = init_params()
    total_loss = []
    beta_k_minus_1 = beta_k
    # Epoch starts from 1
    #data, label = data_iter(group_size)
    data = x_train.loc[1:group_size]
    label = y_train.loc[1:group_size]
    for step in range(2, steps+2):
        output = linear_regression(data, beta_k, beta_0_k)
        loss = obj_func(data, beta_k, group_size, label, lmd)
        beta_update = beta_k + (step - 2)/(step+1) * (beta_k - beta_k_minus_1)
        beta_k_plus_1 = gradient_descent(data, t_k, beta_update, label, group_size)
        loss_plus_1 = obj_func(data, beta_k_plus_1, group_size, label, lmd)
        m_value = m_func(data,beta_k, beta_update, group_size,label, lmd,t_k)
        if loss <= m_value:
            t_k = t_k * 0.5
#         print('losss',loss)
#         print('loss1',loss_plus_1)
        if loss >= loss_plus_1:
            total_loss.append(loss-51.9187)
            beta_k_minus_1 = beta_k
            beta_k = beta_k_plus_1
        else:
            break
    print(loss)
    x_axis = np.linspace(0, steps, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

train(group_size = 100, t_k = 0.005, steps = 5000, lmd = 0.02, n = x_train.shape[0])

def train(group_size, t_k, steps, lmd, n):
    beta_k, beta_0_k = init_params()
    total_loss = []
    # Epoch starts from 1
    #data, label = data_iter(group_size)
    data = x_train.loc[1:group_size]
    label = y_train.loc[1:group_size]
    for step in range(1, steps+1):
        output = linear_regression(data, beta_k, beta_0_k)
        loss = obj_func(data, beta_k, group_size, label, lmd)
        beta_update = gradient_descent(data, t_k, beta_k, label, group_size)
        beta_k_plus_1 = beta_update
        loss_plus_1 = obj_func(data, beta_k_plus_1, group_size, label, lmd)
        m_value = m_func(data, beta_k, beta_update, group_size,label, lmd,t_k)
        if loss <= m_value:
            t_k = t_k * 0.5
#         print('losss',loss)
#         print('loss1',loss_plus_1)
        if loss >= loss_plus_1:
            total_loss.append(loss-51.9187)
            beta_k = beta_k_plus_1
        else:
            break
    print(beta_k)
    print(loss)
    x_axis = np.linspace(0, steps, len(total_loss), endpoint=True)
    plt.semilogy(x_axis, total_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

train(group_size = 200, t_k = 0.005, steps = 5000, lmd = 0.02, n = x_train.shape[0])
