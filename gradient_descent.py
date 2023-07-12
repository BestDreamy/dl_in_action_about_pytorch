# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:55:09 2023

@author: Aric
"""

import numpy as np
import matplotlib.pyplot as plt

# data basis
x_data = [1, 2, 3]
y_data = [2, 4, 6]

# guess a random
w = 1.0

epsilon = 0.01

cost_l = []
episode_l = []

def forward(x):
    return w * x


def cost(nx, ny):
    avg = 0
    for x, y in zip(nx, ny):
        y_pred = forward(x)
        avg += (y_pred - y) ** 2
    return avg / len(nx)

    
def grad(nx, ny):
    ans = 0 # f'(x)
    for x, y in zip(nx, ny):
        ans = ans + (w * x - y) * x
    return ans * 2 / len(nx)

    
def dl():
    for episode in np.arange(1, 101, 1):
        avg = cost(x_data, y_data)
        global w
        w = w - epsilon * grad(x_data, y_data)
        episode_l.append(episode)
        cost_l.append(avg)
        
    
def show():
    plt.plot(episode_l, cost_l, 'y')
    plt.xlabel("episode")
    plt.ylabel("cost")
    plt.show()
    
if __name__ == "__main__":
    dl()
    show()
    print(w)