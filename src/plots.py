import matplotlib.pyplot as pt
import numpy as np 
import algorithms as alg

def plot_cost( cost_history, iterations ):
  pt.plot(np.arange(iterations), cost_history)
  pt.xlabel('# of Iterations')
  pt.ylabel('Cost')
  pt.text(iterations, cost_history[-1], "{0:9.3f}".format(cost_history[-1]))
  pt.show()
  return


def plotLearningCurve(X, y, Xval, yval, l):
  train_err = np.zeros(y.shape)
  valid_err = np.zeros(y.shape)

  for i in range(y.shape[0]):
    theta = alg.parametrize_linear(X[0:i+1, :], y[0:i+1], 0)
    train_err[i] = alg.SSD(theta, X[0:i+1], y[0:i+1], 0)
    valid_err[i] = alg.SSD(theta, Xval, yval, 0)
  
  pt.plot(np.arange(1, train_err.shape[0]+1), train_err, label='Training')
  pt.plot(np.arange(1, valid_err.shape[0]+1), valid_err, label='Validation')
  pt.xlabel('# of Training Examples')
  pt.ylabel('Error')
  pt.legend()
  pt.show()

  return
 