import pandas as pd
import numpy as np

def get_matrices( path_to_data, normalize=False ):
  X, y = read_csv(path_to_data)
  mu = None
  sigma = None
  if(normalize):
    X, mu, sigma = normalize_features(X)
  X = create_design(X)
  theta = np.zeros((X.shape[1], 1))

  return X, y, theta, mu, sigma

def read_csv( path ):
  data = pd.read_csv(path).values
  X = data[:, 0:-1]
  y = data[:, -1].reshape((-1, 1))
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
  assert(X.shape[1]+1 == theta.shape[0])

  if(mu is not None and sigma is not None):
    X_norm = np.zeros(X.shape)
    for i in range(X.shape[1]):
      X_norm[:, i] = ((X[:, i] - mu[0, i]) / sigma[0, i])
    X = X_norm

  return create_design(X) @ theta

