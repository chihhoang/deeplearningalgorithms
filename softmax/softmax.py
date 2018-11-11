import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    results = []
    expValues = np.exp(L)
    
    for l in L:
        results.append(np.exp(l) / np.sum(expValues))
        
    return results