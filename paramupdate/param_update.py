# coding: utf-8
import sys, os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))  # 将根目录加入 Python 路径
sys.path.append(os.pardir)  
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *


def f(x, y):
    return x**2 / 20.0 + y**2


def df(x, y):
    return x / 10.0, 2.0*y

init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0



optimizer = Adam(lr=0.3)
x_history = []
y_history = []
params['x'], params['y'] = init_pos[0], init_pos[1]

for i in range(30):
    x_history.append(params['x'])
    y_history.append(params['y'])
    
    grads['x'], grads['y'] = df(params['x'], params['y'])
    optimizer.update(params, grads)

x = np.arange(-10, 10, 0.01)
y = np.arange(-5, 5, 0.01)

X, Y = np.meshgrid(x, y) 
Z = f(X, Y)

# for simple contour line  
mask = Z > 7
Z[mask] = 0

# plot 

plt.plot(x_history, y_history, 'o-', color="red")
plt.contour(X, Y, Z)
plt.ylim(-10, 10)
plt.xlim(-10, 10)
plt.plot(0, 0, '+')
plt.title("Adam")
plt.xlabel("x")
plt.ylabel("y")
    
plt.show()