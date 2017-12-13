from __future__ import division
import numpy as np
from numpy import matrix
from cconv import *


""" hdmUOG is a variant of hemUOG designed for Holographic Declarative Memory
   Unlike hemUOG, hdmUOG is designed to ONLY include UOGs
   that include the designated item p, where p is an index

   In the HDM model, p is the placeholder vector.

   UOG or Unconstrained Open n-Grams are all ordered sequences of
   characters/vectors in a string, up to grams of numOfItems characters,
   and including sequences that "skip" characters, i.e.,
   sequences are not restricted by adjacency.

   e.g., all UOG of the sequence 'abc' will be:
   a, b, c, ab, ac, bc, abc.

   percepts: matrix of environmental vectors of dimensions [numOfItems,N]
             where: numOfItems is the number of vectors
                    N is the dimensionality of each vector

   chunk (output): sum of all UOG of the environmental vectors in percepts

   p: the index of the placeholder vector, i.e., the item that must be
      included in all skip-grams / UOGs
          
   left: permutation for indicating that a vector is the left operand of
         a non-commutative circular convolution. By default, the function
         uses no permutation (i.e., the numbers 1 to N in ascending order) """



def hdmUOG( percepts, p, left):

    [numOfItems,N] = np.shape(percepts)[0], np.shape(percepts)[1]
    
    if left is None:
		left = range(0,N)

    chunk = [0]*N
    sum = [0]*N
    for i in range (0,numOfItems):
        if i == 0:
            sum = percepts[i] ## row i
        elif (i > 0) and (i < p):
            leftOperand = np.add(chunk, sum)
            chunk = np.add(chunk, cconv(leftOperand[left],percepts[i])) ### leftOperand(left) permutation change
            sum = np.add(sum, percepts[i])
        elif i == p:  # force all skip grams to include item p
            leftOperand = np.add(chunk, sum)
            chunk = cconv(leftOperand[left],percepts[i])
            sum = percepts[i]
        else: #% i > p, i > 1
            leftOperand = np.add(chunk, sum)
            chunk = np.add(chunk, cconv(leftOperand[left],percepts[i]))
    return chunk

