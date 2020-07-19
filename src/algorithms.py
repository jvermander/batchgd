import pandas as pd
import numpy as np

def batch_gd( X, y, theta, alpha, iterations, gradient, l=0 ):
  assert(callable(gradient))
  assert(theta.ndim == 1)  
  
  for i in range(iterations):
    theta -= alpha * gradient(theta, X, y, l)

  return

def batch_gd_debug( X, y, theta, alpha, iterations, gradient, cost, l=0 ):
  assert(callable(cost))
  assert(callable(gradient))
  assert(theta.ndim == 1)

  cost_history = np.zeros((iterations))

  for i in range(iterations):
    theta -= alpha * gradient(theta, X, y, l)
    cost_history[i] = cost(theta, X, y, l)
  
  return cost_history

def normal_eqn( X, y ):
  return np.linalg.pinv(X.T @ X) @ X.T @ y

# Linear Regression cost function: 
# Sum of Squared Differences

def SSD( theta, X, y, l=0 ):
  assert(theta.ndim == 1)

  m = y.shape[0]
  return np.sum(((X @ theta - y) ** 2)) / (2 * m) + reg_cost(theta, l, m)

def SSD_gradient( theta, X, y, l=0 ):
  assert(theta.ndim == 1)

  m = y.shape[0]
  result = X.T @ (X @ theta - y) / m + reg_gradient(theta, l, m)
  return result

# Logistic Regression cost function:
# Cross-Entropy / Log Loss

def cross_ent( theta, X, y, l=0 ):
  assert(theta.ndim == 1)

  m = y.shape[0]
  cost = y * np.log(sigmoid(X @ theta))
  cost += (np.ones(y.shape) - y) * np.log(1 - sigmoid(X @ theta))
  cost = (-1 / m) * np.sum(cost)
  cost += reg_cost(theta, l, m)
  return cost

  # return (-1 / m) * np.sum((y * np.log(sigmoid(X @ theta)) + (np.ones(y.shape) - y) * np.log(1 - sigmoid(X @ theta))))

def cross_ent_gradient( theta, X, y, l=0 ):
  assert(theta.ndim == 1)

  m = y.shape[0]
  return (X.T @ (sigmoid(X @ theta) - y)) / m + reg_gradient(theta, l, m)

def sigmoid( z ):
  return 1 / (1 + np.exp(-z))
  
def reg_cost( theta, l, m ):
  return (l / 2 / m) * (sum(theta ** 2) - theta[0] ** 2)

def reg_gradient( theta, l, m ):
  result = (l / m) * theta
  result[0] -= (l / m) * theta[0]
  return result

