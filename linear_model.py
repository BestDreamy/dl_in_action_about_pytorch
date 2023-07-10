# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:10:51 2023

@author: Aric
"""

import numpy as np
import matplotlib.pyplot as plt


x_data = [1, 2, 3]
y_data = [2, 4, 6]

def forward(x, w):
    return x * w

def loss(y_pred, y):
    return (y_pred - y) ** 2

w_l = []
mse_l = [] # mean square evaluate

def dl():
    for w in np.arange(0.0, 4.1, 0.1):
        avg = 0
        for x, y in zip(x_data, y_data):
            y_pred = forward(x, w)
            avg += loss(y_pred, y) # loss = (y' - y) ^ 2 
        avg /= 3
        w_l.append(w)
        mse_l.append(avg)        
    
def show():
    plt.plot(w_l, mse_l, 'y')
    plt.xlabel("w")
    plt.ylabel("mse")
    plt.show()
    
if __name__ == "__main__":
    dl()
    show()