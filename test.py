import torch
import torch.nn as nn
import math
import numpy as np
from matplotlib import pyplot as plt

dtype = torch.float
device = torch.device("cpu")

X = torch.linspace(1, 100, steps=100, device=device, dtype=dtype).unsqueeze(1) #1-100
Y = torch.linspace(1, 1000, steps=100, device=device, dtype=dtype).unsqueeze(1) # 1-1000

def forward(x):
    return x * w

def criterion(y_pred, y):
    return torch.mean((y_pred - y) ** 2)

# Weight should end up at about 10 since X to Y currently is at a 1:10 ratio    
w = torch.tensor(68, requires_grad=True, dtype=torch.float)

step_size = 0.0001
loss_list = []
w_list = []
iter = 50

for i in range(iter):
    Y_pred = forward(X[i])
    
    loss = criterion(Y_pred, Y[i])
    
    loss_list.append(loss.item())
    
    loss.backward()

    w.data = w.data - step_size * w.grad.data
    
    w.grad.data.zero_()
    w_list.append(w.item())
    
    print('Iter: {}, loss: \t{}, weight: \t{}'.format(i, loss.item(), w.item()))
           
fig, axs = plt.subplots(2, figsize=(6,8))      
axs[0].set_title('Loss change')
axs[0].plot(loss_list, 'r')

axs[1].set_title('Weight change')
axs[1].plot(w_list, 'b')
plt.show()

# Put a random prediction here for end results 69 should be about 690 with the existing data             
print("finished, prediction:", forward(69), )