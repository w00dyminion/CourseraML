import numpy as np
import matplotlib.pyplot as plt


# Simple Function

def warmUpExercise():
    return(np.identity(5))

print(warmUpExercise())

# Univariate linear regression
## Plotting initial data
    
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = np.c_[data[:,0]]
y = np.c_[data[:,1]]

plt.scatter(X,y,s=60,c='r',marker='x',linewidths=1)
plt.xlim(4,24)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.xticks(np.arange(5,21,step=5))

## Gradient Descent
Xtemp = np.c_[np.ones(X.shape[0]), X]
m = y.size


def computeCost(X, y, theta):
    J = (1/(2*m))*np.sum(((Xtemp@theta)-y)**2)
    return(J)
    
theta = np.array([[0],[0]])
print(computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        theta = theta - alpha*(1/m)*Xtemp.T@(Xtemp@theta - y)
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
        
alpha = 0.01
iterations = 1500
theta, J_cost = gradientDescent(X,y,theta,alpha,iterations)
print('theta: ', theta)
plt.plot(X,Xtemp@theta)
plt.legend(['Linear regression', 'Training data'])
plt.show()

plt.plot(J_cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

predict1 = [1, 3.5]@theta
print('For population = 35,000, we predict a profit of ', predict1*10000)
predict2 = [1, 7]@theta
print('For population = 70,000, we predict a profit of ', predict2*10000)