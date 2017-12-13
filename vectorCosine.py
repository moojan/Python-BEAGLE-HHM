import numpy as np
import math

""" VECTOR COSINE computes the cosine of the angle between the vectors x an y
    outputing a value between 1 and -1.
    The cosine is useful as a measure of similarity:
    0 means the vector are orthogonal, or completely dissimilar
    +1 means the vectors are identical
    -1 means the vectors are exact opposites"""

def vectorCosine(x,y):
    xySum=np.dot(x,y)
    xSum=np.dot(x,x)
    ySum=np.dot(y,y)
    return xySum/(math.sqrt(xSum)*math.sqrt(ySum))
    



