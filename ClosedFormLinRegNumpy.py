import numpy as np
import pandas as pd

n = 10
p = 2

x0 = np.ones(n)
x1 = np.random.normal(20,1,n)
x2 = np.random.normal(30,1,n)
X = np.column_stack((x0,x1,x2))
b = np.arange(1,3+1)
b.reshape(3,1)
y =  X @ b

def ClosedFormLinRegNumpy(X,y):
    return np.linalg.inv(X.transpose() @ X) @ X.transpose() @ y
    
bhat = ClosedFormLinRegNumpy(X,y)
print(bhat)