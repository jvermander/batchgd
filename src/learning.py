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

  ex1()
  ex2()
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

def linreg_gd( csv, alpha, iterations, l, normalize=False ):
  X, y, theta, mu, sigma = ut.get_matrices(csv, normalize)
  J = alg.batch_gd_debig(X, y, theta, alpha, iterations, alg.SSD_gradient, alg.SSD, l)
  print("Theta =\n", theta)

  print("Optimal theta = \n", alg.normal_eqn(X, y))

  pt.plot_cost(J, iterations)  
  return theta, mu, sigma

def logreg_gd( csv, alpha, iterations, l, normalize=False ):
  X, y, theta, mu, sigma = ut.get_matrices(csv, normalize)
  X = ut.add_features(X[:, 1], X[:, 2], 6)
  theta = np.zeros((X.shape[1], 1))

  J = alg.batch_gd(X, y, theta, alpha, iterations, alg.cross_ent_gradient, alg.cross_ent, 10)
  print("Theta =\n", theta)

  pt.plot_cost(J, iterations)
  p = np.round(alg.sigmoid(X @ theta))
  print(np.mean(np.equal(p, y)) * 100)
  return theta, mu, sigma

if(__name__ == "__main__"):
  main(sys.argv[1:])


  #   alpha = 0.1
  # iterations = 400
  # theta, mu, sigma = linreg_gd(csv, alpha, iterations, True)

  # while(True):
  #   X = np.zeros((1, theta.shape[0]-1))
  #   for i in range(X.shape[1]):
  #     val = input(">> ")
  #     if(val == 'q'):
  #       return
  #     else:
  #       val = float(val)
  #     X[0, i] = val
  #   print("Predictions =\n", ut.predict(X, theta, mu, sigma))