import numpy as np
import random as rand
import math
import scipy.optimize as opt

opt.minimize

class Neural:

  # for classifications in (1 ... degree)
  def binarize_ground_truth( y, degree ):
    assert(isinstance(y, (np.ndarray)))
    assert(y.ndim == 1)

    result = np.zeros((y.shape[0], degree))
    for i in range(1,degree+1):
      result[:, i-1] = (y == i)
    return result


  def binary_cross_entropy_deriv( x ):
    return

  def sigmoid( x ):
    return 1 / (1 + np.exp(-x))

  def sigmoid_deriv( x ):
    return
    
  def init_random( rows, cols ):
    epsilon = math.sqrt(6) / math.sqrt(rows + cols)
    return [ rand.uniform(0, 1) * 2 * epsilon - epsilon for i in range(rows * cols) ]

  MAX_LAYERS = 10
  MIN_LAYERS = 2

  BCE = { 'f' : binary_cross_entropy, 'deriv' : binary_cross_entropy_deriv }
  SIG = { 'f' : sigmoid, 'deriv' : sigmoid_deriv, 'init' : init_random }

  COST = { 'BCE' : BCE }
  ACT = { 'SIG' : SIG }
  INIT = { 'SIG' : init_random }

  def __init__( self, layers, X, y, l=0, cost='BCE', act='SIG' ):
    assert(isinstance(layers, (np.ndarray)))
    assert(np.issubdtype(layers.dtype, int))
    assert(layers.ndim == 1)
    assert(layers.shape[0] >= Neural.MIN_LAYERS)
    assert(layers.shape[0] <= Neural.MAX_LAYERS)

    self.cost = Neural.COST[cost]
    self.act = Neural.ACT[act]

    self.layer_size = layers
    self.num_layers = layers.shape[0]

    self.y = y.T               # ground truth
    self.m = self.y.shape[1]   # training set size
    self.l = l                 # regularization constant

    self.weight = []
    self.bias = []

    for l in range(1, self.num_layers):
      self.weight += self.act['init'](self.layer_size[l], self.layer_size[l-1])
      self.bias += self.act['init'](self.layer_size[l], 1)
    
    self.weight = np.array(self.weight)
    self.bias = np.array(self.bias)

    # all layer activations for each training example, including the input matrix
    self.a = np.zeros(self.m * (X.shape[1] + self.bias.shape[0]))
    self.a[:X.shape[1] * self.m] = X.T.flatten() 

  def fp( self ):
    weight_start = weight_end = \
    bias_start = bias_end = \
    act_start = act_end = 0
    
    for l in range(1, self.num_layers):
      weight_end += self.layer_size[l] * self.layer_size[l-1]
      bias_end += self.layer_size[l]
      act_end += self.layer_size[l-1] * self.m

      W = self.weight[weight_start : weight_end].reshape(self.layer_size[l], self.layer_size[l-1])
      b = self.bias[bias_start : bias_end].reshape(self.layer_size[l], 1)
      a = self.a[act_start : act_end].reshape(self.layer_size[l-1], self.m)

      self.a[act_end : act_end + self.layer_size[l] * self.m] = self.act['f'](W.dot(a) + b).flatten()

      weight_start = weight_end
      bias_start = bias_end
      act_start = act_end

    degree = self.layer_size[self.num_layers-1]
    return self.a[-(degree * self.m):].reshape(degree, self.m)


  def binary_cross_entropy( self ):
    J = (-1 / self.m) * \
        np.sum(self.y * np.log(self.fp()) +
        (1 - self.y) * np.log(1 - self.fp()))

    J += self.l / 2 / self.m * np.sum(self.weight) ** 2     
    return J  

