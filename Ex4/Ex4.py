import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

#%% Neural Networks
def addThetaZero(X):
    return(np.c_[np.ones(X.shape[0]), X])
    
def sigmoid(X):
    return(1/(1+np.exp(-X)))
   
def sigmoidGradient(X):
    g = sigmoid(X)    
    return(g*(1-g))
#%% # Data Visualisation
data1 = loadmat('ex4data1.mat')
y1 = data1['y']
X1 = data1['X']

sel = np.random.choice(X1.shape[0],20)
plt.imshow(X1[sel,:].reshape(-1,20).T)
plt.axis('off')
plt.show()

#%% # Model Representation
data2 = loadmat('ex4weights.mat')
theta1 = data2['Theta1']
theta2 = data2['Theta2']

#%% # Feedforward and Cost Function
def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambdo):
   Theta1 = nn_params[0:(hidden_layer_size*(input_layer_size + 1))].reshape(hidden_layer_size, input_layer_size+1)
   Theta2 = nn_params[(hidden_layer_size*(input_layer_size + 1)):].reshape(num_labels,hidden_layer_size+1)
   m = X.shape[0]
   J = 0
   Theta1_grad = np.zeros(Theta1.shape)
   Theta2_grad = np.zeros(Theta2.shape)
   Xtemp = addThetaZero(X)
   ytemp = pd.get_dummies(y.ravel()).values
   
   z2 = Xtemp@(Theta1.T)
   a2 = addThetaZero(sigmoid(z2))
   z3 = a2@(Theta2.T)
   a3 = sigmoid(z3)
   
   J = (1/m)*np.sum(-(ytemp*np.log(a3))-((1-ytemp)*np.log(1-a3))) + (lambdo/(2*m))*(np.sum(np.square(Theta1[:,1:]))+np.sum(np.square(Theta2[:,1:])))
   
   part3 = a3 - ytemp 
   part2 = (part3@Theta2[:,1:])*sigmoidGradient(z2)
   
   Theta1_grad = (1/m)*(part2.T@Xtemp) + (lambdo/m)*np.c_[np.zeros(Theta1.shape[0]), Theta1[:,1:]]
   Theta2_grad = (1/m)*(part3.T@a2) + (lambdo/m)*np.c_[np.zeros(Theta2.shape[0]), Theta2[:,1:]]
   grad = np.r_[Theta1_grad.ravel(), Theta2_grad.ravel()]
   return(J,grad)
   
nn_params = np.r_[theta1.ravel(), theta2.ravel()]
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lambdo = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X1, y1, lambdo)[0]
print('no reg', J)

#%% # Regularised Cost Function
lambdo = 1
J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,num_labels, X1, y1, lambdo)[0]
print('reg', J)

#%% Backpropagation
    # Sigmoid Gradient

    
    # Random Initialisation
def randInitializeWeights(inSize, outSize, randCoeff):
    return(np.random.rand(inSize,outSize)*2*randCoeff - randCoeff)

#%% # Backpropagation
