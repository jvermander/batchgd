import matplotlib.pyplot as pt
import numpy as np 

def plot_cost( cost_history, iterations ):
  pt.plot(np.arange(iterations), cost_history)
  pt.xlabel('# of Iterations')
  pt.ylabel('Cost')
  pt.text(iterations, cost_history[-1], "{0:9.3f}".format(cost_history[-1]))
  pt.show()
  return