import pandas as pd
import numpy as np

def batch_gd( X, y, theta, alpha, iterations, gradient ):
  assert(callable(gradient))
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)
  
  for i in range(iterations):
    theta -= alpha * gradient(X, y, theta)

  return

def batch_gd( X, y, theta, alpha, iterations, gradient, cost ):
  assert(callable(cost))
  assert(callable(gradient))
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  cost_history = np.zeros((iterations))

  for i in range(iterations):
    theta -= alpha * gradient(X, y, theta)
    cost_history[i] = cost(X, y, theta)
  
  return cost_history
  

def normal_eqn( X, y ):
  return np.linalg.pinv(X.T @ X) @ X.T @ y

# Linear Regression Cost Function: 
# Sum of Squared Differences

def SSD( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  return np.sum(((X @ theta - y) ** 2)) / (2 * m)

def SSD_gradient( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  return X.T @ (X @ theta - y) / m 






