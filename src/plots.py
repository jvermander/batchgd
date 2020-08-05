import matplotlib.pyplot as pt
import numpy as np 
import algorithms as alg
import utils as ut
import warnings
warnings.filterwarnings("ignore")

def plot_cost( cost_history, iterations ):
  pt.plot(np.arange(iterations), cost_history)
  pt.xlabel('# of Iterations')
  pt.ylabel('Cost')
  pt.text(iterations, cost_history[-1], "{0:9.3f}".format(cost_history[-1]))
  pt.show()
  return

def plot_randomized_learning_curve( X, y, Xval, yval, l, iter=50 ):
  train_err = np.zeros(y.shape)
  valid_err = np.zeros(y.shape)

  for i in range(y.shape[0]):
    for j in range(iter):
      idx = np.random.randint(0, high=y.shape[0], size=i+1)
      theta = alg.parametrize_linear(X[idx, :], y[idx], l)
      train_err[i] += alg.SSD(theta, X[idx, :], y[idx], 0)
      valid_err[i] += alg.SSD(theta, Xval[idx, :], yval[idx], 0)
    train_err[i] /= iter
    valid_err[i] /= iter

  pt.plot(np.arange(1, train_err.shape[0]+1), train_err, label='Training')
  pt.plot(np.arange(1, valid_err.shape[0]+1), valid_err, label='Validation')
  pt.xlabel('# of Training Examples')
  pt.ylabel('Error')
  pt.legend()
  pt.show()
  return


def plot_learning_curve( X, y, Xval, yval, l ):
  train_err = np.zeros(y.shape)
  valid_err = np.zeros(y.shape)

  for i in range(y.shape[0]):
    theta = alg.parametrize_linear(X[0:i+1, :], y[0:i+1], l)
    train_err[i] = alg.SSD(theta, X[0:i+1], y[0:i+1], 0)
    valid_err[i] = alg.SSD(theta, Xval, yval, 0)

  pt.plot(np.arange(1, train_err.shape[0]+1), train_err, label='Training')
  pt.plot(np.arange(1, valid_err.shape[0]+1), valid_err, label='Validation')
  pt.xlabel('# of Training Examples')
  pt.ylabel('Error')
  pt.legend()
  pt.show()

  return

def plot_validation_curve( X, y, Xval, yval ):
  l_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 2.5, 3, 3.3, 10])
  train_err = np.zeros(l_vec.shape)
  valid_err = np.zeros(l_vec.shape)

  for i in range(l_vec.shape[0]):
    theta = alg.parametrize_linear(X, y, l_vec[i])
    train_err[i] = alg.SSD(theta, X, y, 0)
    valid_err[i] = alg.SSD(theta, Xval, yval, 0)

    if(i == 0): print("Lambda\t\tTraining Error\tValidation Error")
    print("%f\t%f\t%f" % (l_vec[i], train_err[i], valid_err[i]))

  pt.plot(l_vec, train_err, label='Training')
  pt.plot(l_vec, valid_err, label='Validation')
  pt.legend('Training', 'Validation')
  pt.xlabel('Lambda')
  pt.ylabel('Error')
  pt.legend()
  pt.show()

  l = l_vec[np.argmin(valid_err)]
  return l

def fit_plot( X, y, mu, sigma, theta, p ):
  pt.plot(X, y, 'rx')
  x = np.arange(X.min() - 10, X.max() + 20, 0.05).reshape(-1, 1)
  X_poly = ut.poly_features(x, p)
  X_poly -= mu
  X_poly = np.divide(X_poly, sigma)
  X_poly = ut.create_design(X_poly)
  pt.plot(x, X_poly @ theta.reshape(-1, 1),  'c--')
  pt.show()
  return
 