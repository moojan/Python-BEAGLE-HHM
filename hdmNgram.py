from __future__ import division
import numpy as np
from numpy import matrix
from cconv import *
from copy import copy

"""hdmNgram produces all the convolutional n-grams of row vectors in percepts
   that contain vector p
   n-grams vary in size from 1 to numOfItems   
   Contrast with hdmUOG, which gets all unconstrained open
   grams of row vectors in percepts that contain vector p.

   percepts: matrix of environmental vectors of dimensions [numOfItems,N]
             where: numOfItems is the number of vectors
                    N is the dimensionality of each vector

   chunk: sum of all n-grams of the environmental vectors in percepts
          
   p: the index of the placeholder vector, i.e., the item that must be
      included in all n-grams
          
   left: permutation for indicating that a vector is the left operand of
         a non-commutative circular convolution. By default, the function
         uses no permutation (i.e., the numbers 1 to N in ascending order) """

def hdmNgram (percepts2, p, left2):
    percepts=copy(percepts2)
    left=copy(left2)
    [numOfItems,N] = np.shape(percepts)[0], np.shape(percepts)[1]
    if left is None:
		left = range(0,N)

    gram = percepts[0]
    sum  = [0]*N
    for i in range (1,numOfItems):
        if i < p: # build up grams
            gram = np.add(percepts[i], cconv(gram[left],percepts[i])) ####
        elif i == p: # begin add grams to the sum now that we've hit p
            gram = cconv(gram[left], percepts[i]) ######## permutation
            sum  = gram
            gram = np.add(gram, percepts[i])
        else: # i > p
            gram = cconv(gram[left],percepts[i]) ## N
            sum  = np.add(sum, gram)
    return sum


