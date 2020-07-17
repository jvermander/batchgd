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

# Linear Regression cost function: 
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


# Logistic Regression cost function:
# Cross-Entropy / Log Loss

def cross_ent( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  cost = y * np.log(sigmoid(X @ theta))
  cost += (np.ones(y.shape) - y) * np.log(1 - sigmoid(X @ theta))
  cost = (-1 / m) * np.sum(cost)
  return cost

  # return (-1 / m) * np.sum((y * np.log(sigmoid(X @ theta)) + (np.ones(y.shape) - y) * np.log(1 - sigmoid(X @ theta))))

def cross_ent_gradient( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  return (X.T @ (sigmoid(X @ theta) - y)) / m

def sigmoid( z ):
  return 1 / (1 + np.exp(-1 * z))
  






