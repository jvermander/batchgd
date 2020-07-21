import sys, getopt

import pandas as pd
import numpy as np
import scipy.optimize as opt

import utils as ut
import algorithms as alg
import plots as pt

np.set_printoptions(edgeitems=5)
np.core.arrayprint._line_width = 180

def main(argv):
  # ex1()
  # ex2()
  ex3()
  return

def ex1():
  X, y = ut.read_csv('csv/ex1data1.csv')
  X = ut.create_design(X)
  theta = np.zeros((X.shape[1],))
  iterations = 1500
  alpha = 0.01
  print("Cost 1 =", alg.SSD(theta, X, y))
  print("Cost 2 =", alg.SSD(np.array([-1, 2]), X, y))
  alg.batch_gd(X, y, theta, alpha, iterations, alg.SSD_gradient)
  print("Theta = \n", theta)
  print("Predict 1 =", ut.predict(np.array([[1, 6.1101]]), theta) * 10 ** 4)
  print("Predict 2 =", ut.predict(np.array([[1, 7]]), theta) * 10 ** 4)
  
  theta = np.zeros((X.shape[1]),)
  res = opt.minimize(alg.SSD, theta, (X, y), jac=alg.SSD_gradient, method='Newton-CG', options={"maxiter": 1500})
  print("Theta = \n", res.x)
  print("Predict 1 =", ut.predict(np.array([[1, 6.1101]]), res.x) * 10 ** 4)
  print("Predict 2 =", ut.predict(np.array([[1, 7]]), res.x) * 10 ** 4)

  print("Data2:")
  X, y = ut.read_csv('csv/ex1data2.csv')
  X, mu, sigma = ut.normalize_features(X)
  X = ut.create_design(X)
  alpha = 0.1
  iterations = 400
  theta = np.zeros((X.shape[1],))
  alg.batch_gd(X, y, theta, alpha, iterations, alg.SSD_gradient)
  print("Theta = \n", theta)

  X, y, = ut.read_csv('csv/ex1data2.csv')
  X = ut.create_design(X)
  alg.normal_eqn(X, y)
  print("Theta = \n", theta)
  return

def ex2():
  X, y = ut.read_csv('csv/ex2data1.csv')
  X = ut.create_design(X)
  theta = np.zeros((X.shape[1],))
  cost = alg.cross_ent(theta, X, y)
  grad = alg.cross_ent_gradient(theta, X, y)
  print("Cost =", cost)
  print("Gradient = \n", grad)
  res = opt.minimize(alg.cross_ent, theta, (X, y), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 400})
  print(res)
  theta = res.x
  p = alg.sigmoid(np.array([1, 45, 85]) @ theta)
  print("Probability =", p)
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print("Training Accuracy =", p)

  X, y = ut.read_csv('csv/ex2data2.csv')
  X = ut.add_features(X[:, 0], X[:, 1], 6)
  print(X)
  theta = np.zeros((X.shape[1],))
  l = 1
  print("Regularized Cost =", alg.cross_ent(theta, X, y, l))
  print("Regularized Gradient =\n", alg.cross_ent_gradient(theta, X, y, l))
  res = opt.minimize(alg.cross_ent, theta, (X, y, l), 
               method='BFGS', 
               jac=alg.cross_ent_gradient,
               options={'maxiter': 1000})
  print(res)
  theta = res.x
  p = np.mean(np.round(alg.sigmoid(X @ theta)) == y) * 100
  print("Training Accuracy =", p)

def ex3():
  X, y = ut.read_mat('mat/ex3data1.mat')
  for i in range(y.shape[0]):
    if(y[i] == 10): y[i] = 0;

  theta = np.array([-2, -1, 1, 2])
  X_t = ut.create_design(np.arange(1, 16, 1).reshape(3, 5).T/10)
  y_t = np.array(([1, 0, 1, 0, 1]))
  l_t = 3
  cost = alg.cross_ent(theta, X_t, y_t, l_t)
  grad = alg.cross_ent_gradient(theta, X_t, y_t, l_t)
  print("Expected / Actual:")
  print("2.534819 / %f" % cost)
  print("0.146561 / %f" % grad[0])
  print("-0.548558 / %f" % grad[1])
  print("0.724722 / %f" % grad[2])
  print("1.398003 / %f" % grad[3])
  
  degree = 10
  l = 0.1
  theta = ut.multiclass_logreg(X, y, l, degree)
  p = ut.multiclass_prediction(theta, X)
  print(p.shape)
  print("95 / %f" % (np.mean(p == y) * 100))
  return


if(__name__ == "__main__"):
  main(sys.argv[1:])