import numpy as np
import pandas as pd

def init_centroids( X, K ):
  rand = np.random.randint(0, X.shape[0], K)
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
  return mu

def compute_cost( X, mu, idx ):
  mapping = mu[idx]

  c = np.sum((X - mapping) ** 2, axis=None) / X.shape[0]
  return c