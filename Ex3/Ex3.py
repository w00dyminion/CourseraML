import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize

# Multi-class Classification
def addThetaZero(X):
    Xtemp = np.c_[np.ones(X.shape[0]), X]
    return(Xtemp)

## Dataset    
data1 = loadmat('ex3data1.mat')
y1 = data1['y']
X1 = data1['X']
X1temp = addThetaZero(X1)
[m,n] = np.shape(X1temp)

## Visualistaion
sel = np.random.choice(m,20)
plt.imshow(X1[sel,:].reshape(-1,20).T)
plt.axis('off')
plt.show()

## Vectorising regression
def sigmoid(z):
    return(1/(1+np.exp(-z)))

def lrcostFunctionReg(theta_init,X,y,lambdo):
    theta = theta_init.reshape((theta_init.shape[0],1))
    J = 0
    grad = np.zeros(theta.shape)
    thmod = grad + 1
    thmod[0,0] = 0
    h = sigmoid(X@theta)
    thsq = theta*thmod
    J = (1/m)*((-y.T@np.log(h))-((1-y).T@np.log(1-h)))+(lambdo/(2*m)*sum(thsq**2))
    grad = (1/m)*X.T@(h-y)+(lambdo/m)*thsq
    return(J[0],grad.flatten())
    
thetaTest = np.array([-2,-1,1,2])
Xtest = addThetaZero(((np.reshape(np.arange(15),(3,5))+1)/10).T)
[m,n] = np.shape(Xtest)
ytest = np.array([[1],[0],[1],[0],[1]])
lambdotest = 3
J, grad = lrcostFunctionReg(thetaTest, Xtest,ytest,lambdotest)
print(J)
print(grad)

## One vs all
[m,n] = np.shape(X1temp)

a = np.arange(10)
b = 3
print((a==b).astype(int))

