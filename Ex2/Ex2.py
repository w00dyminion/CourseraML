import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
np.seterr(divide='ignore')

# Logistic Regression
def addThetaZero(X):
    Xtemp = np.c_[np.ones(X.shape[0]), X]
    return(Xtemp)
    
data1 = np.loadtxt('ex2data1.txt', delimiter = ',')
X1 = np.c_[data1[:,np.arange(data1.shape[1]-1)]]
y1 = np.c_[data1[:,(data1.shape[1]-1)]]
X1temp = addThetaZero(X1)
[m,n] = np.shape(X1temp)

## Data visualisation
[positives,NULL] = np.where(y1==1)
[negatives,NULL] = np.where(y1==0)
plt.scatter(X1[positives,0],X1[positives,1],marker='+')
plt.scatter(X1[negatives,0],X1[negatives,1],c='y')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(['Admitted','Not Admitted'])
plt.show()

# Implementaion
## Sigmoid function
def sigmoid(z):
    g = np.zeros(z.shape)
    g = 1/(1+np.exp(-z))
    return g

print('sigmoid(7) =', sigmoid(np.array([[-7]])))

## Cost funtion and gradient
theta1_init = np.zeros((n))
def costFunction(theta_init,X,y):
    theta = theta_init.reshape((3,1))
    J = 0
    grad = np.zeros(theta.shape)
    h = sigmoid(X@theta)
    J = (1/m)*((-y.T@np.log(h))-((1-y).T@np.log(1-h)))
    grad = (1/m)*X.T@(h-y)
    return(J[0],grad.flatten())

J, grad = costFunction(theta1_init,X1temp,y1)    
print('J =',J)
print('grad =',grad)

## Automatic minimisation
optiResult = minimize(costFunction,theta1_init,args=(X1temp,y1),jac=True,options={'maxiter':400})
theta = optiResult['x']
cost = optiResult['fun']
print('Optimal theta =', theta)
print('With cost =', cost)