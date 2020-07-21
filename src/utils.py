import pandas as pd
import numpy as np
import scipy.io as io
import scipy.optimize as opt
import time

import algorithms as alg

def read_csv( path ):
  data = pd.read_csv(path).values
  X = data[:, 0:-1]
  y = data[:, -1]
  return X, y

def read_mat( path ):
  data = io.loadmat(path)
  X = data['X']
  y = data['y'].reshape(-1)
  return X, y

def normalize_features( X ):
  mu = np.zeros((1, X.shape[1]))
  sigma = np.zeros((1, X.shape[1]))
  X_norm = np.zeros(X.shape)

  for i in range(X.shape[1]):
    mu[0, i] = np.mean(X[:, i])
    sigma[0, i] = np.std(X[:, i], ddof=1)

    X_norm[:, i] = ((X[:, i] - mu[0, i]) / sigma[0, i])
  
  return X_norm, mu, sigma

def create_design( X ):
  m = X.shape[0]
  return np.hstack((np.ones((m, 1)), X))
  

def predict( X, theta, mu=None, sigma=None ):

  if(mu is not None and sigma is not None):
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
      X_norm[:, i] = ((X[:, i] - mu[0, i]) / sigma[0, i])
    X = X_norm

  return X @ theta

def add_features( X1, X2, degree ):
  assert(X1.shape == X2.shape)

  columns = (degree * (degree + 1))/2 + degree + 1# n(n+1)/2 + n + 1
  result = np.ones((X1.shape[0],int(columns)))

  k = 1
  for i in range(1, degree+1):
    for j in range(i+1):
      result[:, k] = np.multiply(X1 ** (i-j), X2 ** j)
      k += 1
  assert(k == columns)
  return result

def multiclass_logreg( X, y, l, degree ):
  X = create_design(X)
  theta = np.zeros((degree, X.shape[1]))
  for i in range(degree):
    res = opt.minimize(alg.cross_ent, theta[i, :], 
                    (X, y == i, l), jac=alg.cross_ent_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 50})
    # print(res)
    theta[i, :] = res.x
  return theta

def multiclass_prediction( theta, X ):
  X = create_design(X)
  return np.argmax(theta @ X.T, axis=0)


