import sys, getopt
import warnings

import pandas as pd
import numpy as np
import scipy.optimize as opt

import utils as ut
import algorithms as alg
import plots as pt

np.set_printoptions(edgeitems=5)
np.core.arrayprint._line_width = 180
warnings.filterwarnings("ignore", category=RuntimeWarning) 

def main(argv):

  test1()
  test2()
  test3()
  return

def test1():
  print("\n\nTest 1 - Linear Regression")
  print("Expected / Actual:")
  
  print("\nBatch gradient descent: ")
  X, y = ut.read_csv('csv/ex1data1.csv')
  X = ut.create_design(X)
  theta = np.zeros((X.shape[1],))
  iterations = 1500
  alpha = 0.01
  print("32.0727 / ", alg.SSD(theta, X, y))
  print("52.2425 / ", alg.SSD(np.array([-1, 2]), X, y))
  alg.batch_gd(X, y, theta, alpha, iterations, alg.SSD_gradient)
  print("-3.630291 / ", theta[0])
  print("1.166362 / ", theta[1])
  print("34962.991574 / ", ut.predict(np.array([[6.1101]]), theta)[0] * 10 ** 4)
  print("45342.450129 / ", ut.predict(np.array([[7]]), theta)[0] * 10 ** 4)
  
  print("\nWith optimization: ")
  theta = np.zeros((X.shape[1]),)
  res = opt.minimize(alg.SSD, theta, (X, y), jac=alg.SSD_gradient, method='Newton-CG', options={"maxiter": 1500})
  theta = res.x
  print("-3.630291 / ", theta[0])
  print("1.166362 / ", theta[1])
  print("34962.991574 / ", ut.predict(np.array([[6.1101]]), theta)[0] * 10 ** 4)
  print("45342.450129 / ", ut.predict(np.array([[7]]), theta)[0] * 10 ** 4)
  
  print("\nNormalized batch gradient descent:")
  X, y = ut.read_csv('csv/ex1data2.csv')
  X, mu, sigma = ut.normalize_features(X)
  X = ut.create_design(X)
  alpha = 0.1
  iterations = 400
  theta = np.zeros((X.shape[1],))
  alg.batch_gd(X, y, theta, alpha, iterations, alg.SSD_gradient)
  print("2000.680851 / ", mu[0])
  print("3.170213 / ", mu[1])
  print("794.7024 / ", sigma[0])
  print("0.7610 / ", sigma[1])
  print("340412.659574 / ", theta[0])
  print("110631.048958 / ", theta[1])
  print("-6649.472950 / ", theta[2])

  print("\nNormal equation:")
  X, y, = ut.read_csv('csv/ex1data2.csv')
  X = ut.create_design(X)
  alg.normal_eqn(X, y)
  print("340412.659574 / ", theta[0])
  print("110631.048958 / ", theta[1])
  print("-6649.472950 / ", theta[2])

  print("\nNormalized prediction:")
  print("293081.464622 / ", ut.predict(np.array([[1650, 3]]), theta, mu, sigma)[0])
  print("284343.447245 / ", ut.predict(np.array([[1650, 4]]), theta, mu, sigma)[0])

  return

def test2():
  print("\n\nTest 2 - Logistic Regression & Regularization")
  print("Expected / Actual:")

  print("\nCost & Gradient:")
  X, y = ut.read_csv('csv/ex2data1.csv')
  X = ut.create_design(X)
  theta = np.zeros((X.shape[1],))
  cost = alg.cross_ent(theta, X, y)
  grad = alg.cross_ent_gradient(theta, X, y)
  print("0.693147 / ", cost)
  print("-0.1000 / ", grad[0])
  print("-12.0092 / ", grad[1])
  print("11.2628 / ", grad[2])
  res = opt.minimize(alg.cross_ent, theta, (X, y), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 400})
  print("0.203498 / ", res.fun)
  theta = res.x
  print("-25.1613 / ", theta[0])
  print("0.2062 / ", theta[1])
  print("0.2015 / ", theta[2])
  p = alg.sigmoid(ut.predict(np.array([[45, 85]]), theta)[0])
  print("0.776291 /", p)
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print(">= 89.000000 /", p)

  print("\nRegularization:")
  X, y = ut.read_csv('csv/ex2data2.csv')
  X = ut.add_features(X[:, 0], X[:, 1], 6)
  print("118 / ", X.shape[0])
  print("28 /", X.shape[1])
  print("8.2291e-10 / ", X[117, 27])
  print("0.2914 / ", X[99, 9])
  theta = np.zeros((X.shape[1],))
  l = 1
  print("0.693147 / ", alg.cross_ent(theta, X, y, l))
  grad = alg.cross_ent_gradient(theta, X, y, l)
  print("(28,) / ", grad.shape)
  print("0.0085 / ", grad[0])
  print("0.0129 / ", grad[12])
  print("0.0388 / ", grad[27])
  l = 0
  res = opt.minimize(alg.cross_ent, theta, (X, y, l), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 1000})
  theta = res.x
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print(">= 88.983051 / ", p)
  
  theta = np.zeros((X.shape[1],))
  l = 1
  res = opt.minimize(alg.cross_ent, theta, (X, y, l), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 1000})
  theta = res.x
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print(">= 83.050847 / ", p)
  
  theta = np.zeros((X.shape[1],))
  l = 10
  res = opt.minimize(alg.cross_ent, theta, (X, y, l), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 1000})
  theta = res.x
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print(">= 74.576271 / ", p)
  
  theta = np.zeros((X.shape[1],))
  l = 100
  res = opt.minimize(alg.cross_ent, theta, (X, y, l), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 1000})
  theta = res.x
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print(">= 61.016949 / ", p)

def test3():
  print("\n\nTest 3 - Multiclass Logistic Regression & Neural Networks")
  print("Expected / Actual:")

  print("\nMulticlass LR:")  
  X, y = ut.read_mat('mat/ex3data1.mat')
  for i in range(y.shape[0]):
    if(y[i] == 10): y[i] = 0;

  theta = np.array([-2, -1, 1, 2])
  X_t = ut.create_design(np.arange(1, 16, 1).reshape(3, 5).T/10)
  y_t = np.array(([1, 0, 1, 0, 1]))
  l_t = 3
  cost = alg.cross_ent(theta, X_t, y_t, l_t)
  grad = alg.cross_ent_gradient(theta, X_t, y_t, l_t)
  print("2.534819 / %f" % cost)
  print("0.146561 / %f" % grad[0])
  print("-0.548558 / %f" % grad[1])
  print("0.724722 / %f" % grad[2])
  print("1.398003 / %f" % grad[3])
  
  degree = 10
  l = 0.1
  theta = alg.multiclass_logreg(X, y, l, degree)
  p = ut.multiclass_prediction(theta, X)
  print(">= 95 / %f" % (np.mean(p == y) * 100))
  
  data = ut.read_mat_raw('mat/ex3weights.mat')
  theta1 = data['Theta1']
  theta2 = data['Theta2']
  
  print("\nNeural Networks (Forward Propagation): ")
  X, y = ut.read_mat('mat/ex3data1.mat')
  p = test3neuralnet(theta1, theta2, X)
  print("Predicted: ", p)
  print("Actual: ", y)
  print("Expected vs. Actual Accuracy: 97.52 / %f" % (np.mean(p == y) * 100))
  return

# Forward propagation
def test3neuralnet( theta1, theta2, a_1 ):
  # sizes - layer 1: 400, layer 2: 25, layer 3: 10
  a_1 = ut.create_design(a_1)
  a_2 = alg.sigmoid(a_1 @ theta1.T)
  a_2 = ut.create_design(a_2)
  a_3 = alg.sigmoid(theta2 @ a_2.T)
  p = np.argmax(a_3, axis=0) + 1
  return p

if(__name__ == "__main__"):
  main(sys.argv[1:])