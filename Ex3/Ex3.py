import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

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

def oneVsAll(X,y,num_labels,lambdo):
    all_theta = np.zeros((num_labels,n))
    thetaInit = np.zeros((n,1))
    for c in np.arange(1, num_labels+1):
        out = minimize(lrcostFunctionReg, thetaInit, args=(X,(y==c)*1,lambdo),jac=True, options={'maxiter':50})
        all_theta[c-1] = out.x
    return(all_theta)

num_labels = 10
lambdo = 0.1

all_theta = oneVsAll(X1temp, y1, num_labels, lambdo)
    
def predictOvA(allTheta, X):
    p = sigmoid(X.dot(allTheta.T))
    return(np.argmax(p, axis=1)+1)
    
p = predictOvA(all_theta, X1temp)
print('Training set accuracy: {} %'.format(np.mean(p == y1.ravel())*100))

autoLR = LogisticRegression(C=10)
autoLR.fit(X1,y1.ravel())
p2 = autoLR.predict(X1)
print('Training set accuracy: {} %'.format(np.mean(p2 == y1.ravel())*100))

# Neural Networks
weightData = loadmat('ex3weights.mat')
theta1 = weightData['Theta1']
theta2 = weightData['Theta2']

def predict(Theta1, Theta2, X):
    a2 = sigmoid(X@(Theta1.T))
    a2temp = addThetaZero(a2)
    a3 = sigmoid(a2temp@(theta2.T))
    return(np.argmax(a3, axis=1)+1)
    
p3 = predict(theta1, theta2, X1temp)
print('Training set accuracy: {} %'.format(np.mean(p3 == y1.ravel())*100))

rp = np.random.choice(m,1)
rpred = predict(theta1, theta2, X1temp[rp,:])
print('Neural Network Prediction: {}'.format(rpred.flat[0]))
plt.imshow(X1[rp,:].reshape(-1,20).T)
plt.axis('off')
plt.show()