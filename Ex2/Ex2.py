import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
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

def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))
    
p = predict(theta, X1temp)
print('prediction accuracy = {}%'.format(100*sum(p == y1.ravel())/p.size))

[positives,NULL] = np.where(y1==1)
[negatives,NULL] = np.where(y1==0)
plt.scatter(X1[positives,0],X1[positives,1],marker='+')
plt.scatter(X1[negatives,0],X1[negatives,1],c='y')
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
plt.legend(['Admitted','Not Admitted'])
plt.scatter(45, 85, s=60, c='r', marker='v', label='(45, 85)')
x1_min, x1_max = X1temp[:,1].min(), X1temp[:,1].max(),
x2_min, x2_max = X1temp[:,2].min(), X1temp[:,2].max(),
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
h = sigmoid(np.c_[np.ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta))
h = h.reshape(xx1.shape)
plt.contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
plt.show()


# Regularised regression
## Visualisation
data2 = np.loadtxt('ex2data2.txt', delimiter = ',')
X2 = np.c_[data2[:,np.arange(data2.shape[1]-1)]]
y2 = np.c_[data2[:,(data2.shape[1]-1)]]

[positives2,NULL] = np.where(y2==1)
[negatives2,NULL] = np.where(y2==0)
plt.scatter(X2[positives2,0],X2[positives2,1],marker='+')
plt.scatter(X2[negatives2,0],X2[negatives2,1],c='y')
plt.xlabel('Microchip test 1')
plt.ylabel('Microchip test 2')
plt.legend(['y = 1','y = 0'])
plt.show()

poly = PolynomialFeatures(6)
X2temp = poly.fit_transform(X2)
[m,n] = np.shape(X2temp)

def costFunctionReg(theta_init,X,y,lambdo):
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
    
theta2_init = np.zeros((n))
lambdo = 1
J2, grad2 = costFunctionReg(theta2_init,X2temp,y2,lambdo)    
print('J2 =',J2)
print('grad2 =',grad2)

fig, axes = plt.subplots(1,3, sharey = True, figsize=(17,5))

for i, C in enumerate([0,1,100]):
    optiResult2 = minimize(costFunctionReg, theta2_init, args=(X2temp,y2,C),jac=True, options={'maxiter':3000})
    accuracy = 100*sum(predict(optiResult2.x, X2temp) == y2.ravel())/y2.size
    [positives2,NULL] = np.where(y2==1)
    [negatives2,NULL] = np.where(y2==0)
    axes.flatten()[i].scatter(X2[positives2,0],X2[positives2,1],marker='+')
    axes.flatten()[i].scatter(X2[negatives2,0],X2[negatives2,1],c='y')
    plt.xlabel('Microchip test 1')
    plt.ylabel('Microchip test 2')
    plt.legend(['y = 1','y = 0'])
    x1_min, x1_max = X2temp[:,1].min(), X2temp[:,1].max(),
    x2_min, x2_max = X2temp[:,2].min(), X2temp[:,2].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    h = sigmoid(poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(optiResult2.x))
    h = h.reshape(xx1.shape)
    axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='b')
    axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))