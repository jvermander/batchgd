# Machine Learning:
# Multivariate Linear Regression via
# Batch Gradient Descent & Normal Equation

import pandas as pd
import numpy as np

# For a given hypothesis (theta), 
# computes the average squared error across all training examples (X)

def compute_error( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  return np.sum(((X @ theta - y) ** 2)) / (2 * m)

# Same as compute_error, but without vectorization
# Performs slower, but written out of interest
def compute_error_iter( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  sum = 0
  m = y.shape[0]
  n = X.shape[1]
  for i in range(m):
    temp = 0
    for j in range(n):
      temp +=  X[i, j] * theta[j, 0]
    temp -= y[i, 0]
    sum += temp ** 2
    
  return sum / (2 * m)

# Batch Gradient Descent
def batch_gd ( X, y, theta, alpha, iterations ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)
  
  m = y.shape[0]
  for i in range(iterations):
    theta -= (alpha / m) * X.T @ (X @ theta - y)

  return

# Batch Gradient Descent -- unvectorized (and thus slow), as an exercise
def batch_gd_iter( X, y, theta, alpha, iterations ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  n = theta.shape[0]
  temp = np.zeros(n,)

  for i in range(iterations):
    for j in range(n):

      # compute partial derivative of the error function 
      deriv = 0
      for k in range(m):
        sum = 0
        for l in range(n):
          sum += X[k, l] * theta[l, 0]
        deriv += (sum - y[k, 0]) * X[k, j]
      temp[j] = theta[j, 0] - alpha / m * deriv

    # simultaneous assignment for all parameters
    for k in range(n):
      theta[k, 0] = temp[k]
    
  return

# Equation for solving theta without gradient descent
# Performs slower for large inputs
def normal_eqn( X, y ):
  return np.linalg.pinv(X.T @ X) @ X.T @ y

# Pre-processing optimization for gradient descent
# For each training example, confine each feature to a small range
def normalize_features( X ):
  mu = np.zeros((1, X.shape[1]))
  sigma = np.zeros((1, X.shape[1]))
  X_norm = np.zeros(X.shape)

  for i in range(X.shape[1]):
    mu[0, i] = np.mean(X[:, i])
    sigma[0, i] = np.std(X[:, i], ddof=1)

    X_norm[:, i] = ((X[:, i] - mu[0, i]) / sigma[0, i])
  
  return X_norm, mu, sigma

data = pd.read_csv("csv/ex1data2.csv").values
X = data[:, [0, 1]]
y = data[:, 2].reshape((-1, 1))
n = X.shape[1]
m = y.shape[0]

X, mu, sigma = normalize_features(X)
X = np.hstack((np.ones((m, 1)), X))

theta = np.zeros((3, 1))
batch_gd(X, y, theta, 0.1, 400)
print(theta)

theta = normal_eqn(X, y)
print(theta)

# J = compute_error_iter(X, y, theta)
# print(J)
# J = compute_error(X, y, theta)
# print(J)

# J = compute_error_iter(X, y, np.array([[-1], [2]]))
# print(J)
# J = compute_error(X, y, np.array([[-1], [2]]))
# print(J)

# theta = np.zeros((n + 1, 1))
# batch_gd(X, y, theta, 0.01, 5000)
# print(theta)

# theta = np.zeros((n + 1, 1))
# batch_gd_iter(X, y, theta, 0.01, 5000)
# print(theta)

# theta = normal_eqn(X, y)
# print(theta)

# print(np.array([1, 7]) @ theta * 10 ** 4)

