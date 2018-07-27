import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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
def computeCost(X, y, theta):
    Xtemp = np.c_[np.ones(X.shape[0]), X]
    m = y.size
    J = (1/(2*m))*np.sum(((Xtemp@theta)-y)**2)
    return(J)
    
theta = np.array([[0],[0]])
print(computeCost(X, y, theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    Xtemp = np.c_[np.ones(X.shape[0]), X]
    m = y.size
    J_history = np.zeros(num_iters)
    for iter in np.arange(num_iters):
        theta = theta - alpha*(1/m)*Xtemp.T@(Xtemp@theta - y)
        J_history[iter] = computeCost(X, y, theta)
    return(theta, J_history)
        
alpha = 0.01
iterations = 1500
Xtemp = np.c_[np.ones(X.shape[0]), X]
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


# Visualising J(theta)
t0_vals = np.linspace(-10,10,100)
t1_vals = np.linspace(-1,4,100)
xx, yy = np.meshgrid(t0_vals, t1_vals, indexing='xy')
z = np.zeros((t0_vals.size,t1_vals.size))
for (i,j),v in np.ndenumerate(z):
    z[i,j] = computeCost(X,y, theta=[[xx[i,j]], [yy[i,j]]])
    
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

ax1.contour(xx, yy, z, np.logspace(-2, 3, 20))
ax1.scatter(theta[0],theta[1], c='r')

ax2.plot_surface(xx, yy, z, alpha=0.7, cmap=plt.cm.jet)
ax2.set_zlabel('Cost')
ax2.view_init(elev=15, azim=230)

for ax in fig.axes:
    ax.set_xlabel(r'$\theta_0$', fontsize=17)
    ax.set_ylabel(r'$\theta_1$', fontsize=17)

