
import numpy as np

"""GETSHUFFLE provides a permutation and inverse permutation 
   of the numbers 1:N.
   perm: for shuffling
   invPerm: for unshuffling"""

def getShuffle(n):
    perm = np.random.permutation(n)
    invperm = perm[::-1]
    return perm #, invperm

