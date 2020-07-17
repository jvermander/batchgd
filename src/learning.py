import sys, getopt

import pandas as pd
import numpy as np

import utils as ut
import algorithms as alg
import plots as pt

def main(argv):
  csv = "../csv/ex2data1.csv"
  alpha = 10
  iterations = 100

  theta, mu, sigma = logreg_gd(csv, alpha, iterations, True)

  return

def linreg_gd( csv, alpha, iterations, normalize=False ):
  X, y, theta, mu, sigma = ut.get_matrices(csv, normalize)
  J = alg.batch_gd(X, y, theta, alpha, iterations, alg.SSD_gradient, alg.SSD)
  print("Theta =\n", theta)

  print("Optimal theta = \n", alg.normal_eqn(X, y))

  # pt.plot_cost(J, iterations)  
  return theta, mu, sigma

def logreg_gd( csv, alpha, iterations, normalize=False ):
  X, y, theta, mu, sigma = ut.get_matrices(csv, normalize)
  J = alg.batch_gd(X, y, theta, alpha, iterations, alg.cross_ent_gradient, alg.cross_ent)
  print("Theta =\n", theta)

  pt.plot_cost(J, iterations)
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