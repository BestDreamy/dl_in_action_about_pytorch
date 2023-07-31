# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 19:57:25 2023

@author: Aric
"""

import torch
import numpy as py
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
w = torch.tensor(1.0) # w的初值为1.0
w.requires_grad = True # 需要计算梯度
 
"""
loss = (x * w - y) ^ 2

Use tensor get a grad of (loss)'_w, rather than get it by hand
You can understand this code with gradient_descent algorithm
"""

episode_l = []
w_l = []
gamma = 0.01

def forward(x):
    return x * w
    
    
def loss(x, y):
    pred_y = forward(x)
    # Return a tensor which has elements of (data and grad_fn)
    return (pred_y - y) ** 2 
        
    
def dl():
    for episode in range(100):
        for x, y in zip(x_data, y_data):
            # Get loss = (x * w - y) ^ 2
            # And auto create a compute graph
            l = loss(x, y)
            # Get loss's grad
            l.backward()
            w.data = w.data - gamma * w.grad.data
            w.grad.zero_()
            
            w_l.append(w.item())
            episode_l.append(episode)     
            
    
def dump(t = torch.tensor([[1., 2.], [3., 4.]], requires_grad = True)):
    # t and t.data's type is tensor, but t[1][1].item()'s type is float
    # t.item() must only return a number, rather than a array
    print('--------------------------------------------------------------')
    print('**************************************************************')
    print(f't = {t}\nt.data = {t.data}\nt[1][0].item() = {t[1][0].item()}\nt.grad = {t.grad}')
    print('**************************************************************')
    print('--------------------------------------------------------------')
    
    
def show():
    plt.xlabel('episode')
    plt.ylabel('w')
    plt.plot(episode_l, w_l, 'y')
    plt.show()
    
    
if __name__ == "__main__":
    # dl()
    # show()
    
    """
    Tensor 基本运算:
     1 * 5   2 * 5     10   10     -5   0    ^2      25      0
    10 * 5  20 * 5  -  10   10  =  40  90  ----->  1600   8100 
     5 * 5   5 * 5     10   10     15  15           225    225
     
    The parameter of grad_fn changes as operation changes
    grad_fn  from  MulBackward  --->  SubBackward  --->   PowBackward
    """
    
    """
    Test case about tensor
    refer blog:
        https://www.cnblogs.com/wuxero/p/14138493.html
    -----------------------------------------------------------------------------------------------------------------
    t = torch.tensor([[1., 2.], [10., 20.], [5., 5.]], requires_grad = True)
    t = (t * 5 - 10) ** 2
    # Tensor.backward() just valid for a number, but invaild a vector(matrix)
    # So we need get a sum about this matrix
    t.sum().backward()
    dump(t)
    -----------------------------------------------------------------------------------------------------------------
    """


    """
    Test case about tensor of grad
    refer blog:
        https://blog.csdn.net/m0_46653437/article/details/112912467
    -----------------------------------------------------------------------------------------------------------------
    res = torch.tensor([1.], requires_grad = True)
    print(f'res = {res}\nres.data = {res.data}\nres.grad_fn = {res.grad_fn}\nres.grad = {res.grad}\n')
    ans = res * 2
    print(f'ans = {ans}\nans.data = {ans.data}\nans.grad_fn = {ans.grad_fn}\n')
    ans.backward()
    print(f'res = {res}\nres.data = {res.data}\nres.grad_fn = {res.grad_fn}\nres.grad = {res.grad}\nres.grad.data = {res.grad.data}')
    -----------------------------------------------------------------------------------------------------------------
    """

    res = torch.tensor([[1.], [2.], [3.]], requires_grad = True)
    ans = res * 2
    ans.backward()
    print(f'ans = {ans}, res.grad.data = {res.grad.data}')
