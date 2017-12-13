import numpy as np

#vecNorm normalizes vector "a" to a Euclidean length (i.e., magnitude) of 1

def vecNorm (a):
  return  np.divide(a, np.sqrt(np.dot(a,a)))

