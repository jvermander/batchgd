import numpy as np
import pandas as pd

def init_centroids( X, K ):
  rand = np.random.randint(0, X.shape[0], K)
  # This initialization happens to mess up arithmetic (duplicate points X[10082], X[12942] and X[375]) if bad centroids are not removed from mu
  # rand = np.array([10882, 8063, 12942,  4795, 11177,  1469,  4262, 10635, 14947,  4034,  2950,  5424,   375,  5018, 15865, 15797])
  return X[rand]

def clusterize( X, K, iter ):
  mu = init_centroids(X, K)

  for i in range(iter):
    idx = assign_centroids(X, mu)
    mu = adjust_centroids(X, idx, K)

  idx = assign_centroids(X, mu)
  
  return mu, idx

def assign_centroids( X, mu ):
  idx = np.argmin(np.sum((X[:, np.newaxis] - mu) ** 2, axis=2), axis=1)
  return idx

def adjust_centroids( X, idx, K ):
  mu = np.array([np.mean(X[idx == k], axis=0) for k in range(K)])
  mu = mu[~np.isnan(mu[:, 0])] # discard NaN (centroids without any points assigned to them); 
                               # will mess up arithmetic later if not removed
  return mu

def compute_cost( X, mu, idx ):
  mapping = mu[idx]

  c = np.sum((X - mapping) ** 2, axis=None) / X.shape[0]
  return c