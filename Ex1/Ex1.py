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

# extra work

data2 = np.loadtxt('ex1data2.txt', delimiter = ',')
X2 = np.c_[data2[:,np.arange(data2.shape[1]-1)]]
y2 = np.c_[data2[:,(data2.shape[1]-1)]]

def featureNormalize(a):
    X_norm = np.zeros(X2.shape)
    mu = np.zeros((1,X_norm.shape[1]))
    sigma = np.zeros((1,X_norm.shape[1]))
    for l in np.arange(X_norm.shape[1]):
        mu[0,l] = np.mean(a[:,l])
        sigma[0,l] = np.std(a[:,l])
        X_norm[:,l] = (a[:,l]-mu[0,l])/sigma[0,l]
    return(X_norm, mu, sigma)
    
X_norm, mu, sigma = featureNormalize(X2)
theta_2 = np.zeros((3,1))
alpha_2 = 0.1
num_iters_2 = 400
theta_2,cost2 = gradientDescent(X_norm, y2, theta_2, alpha_2, num_iters_2)
print(theta_2)
price = theta_2[0] + ((1650-mu[0,0])/sigma[0,0])*theta_2[1] + ((3-mu[0,1])/sigma[0,1])*theta_2[2]
print(price)

def normalEqn(X,y):
    Xtemp = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((Xtemp.shape[1],1))
    theta = np.linalg.pinv(Xtemp.T@Xtemp)@Xtemp.T@y
    return(theta)

theta_3 = normalEqn(X2,y2)    
print(theta_3)
price2 = theta_3[0] + 1650*theta_3[1] + 3*theta_3[2]
print(price2)