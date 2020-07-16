# Alternative implementations which are
# not only slower, but also more complex!

def batch_gd( X, y, theta, alpha, iterations ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  m = y.shape[0]
  n = theta.shape[0]
  temp = np.zeros(n,)

  for i in range(iterations):
    for j in range(n):

      # compute partial derivative of the error function 
      deriv = 0
      for k in range(m):
        sum = 0
        for l in range(n):
          sum += X[k, l] * theta[l, 0]
        deriv += (sum - y[k, 0]) * X[k, j]
      temp[j] = theta[j, 0] - alpha / m * deriv

    # simultaneous assignment for all parameters
    for k in range(n):
      theta[k, 0] = temp[k]
    
  return

def SSD( X, y, theta ):
  assert(y.shape[1] == 1)
  assert(theta.shape[1] == 1)

  sum = 0
  m = y.shape[0]
  n = X.shape[1]
  for i in range(m):
    temp = 0
    for j in range(n):
      temp +=  X[i, j] * theta[j, 0]
    temp -= y[i, 0]
    sum += temp ** 2
    
  return sum / (2 * m)