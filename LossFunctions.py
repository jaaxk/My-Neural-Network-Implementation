from math import log

def cross_entropy(pred: list, true: list) -> float:
    if len(pred) != len(true):
        raise IndexError('Output layer must be same size as number of categories')
    
    loss = 0.0
    for p, t in zip(pred, true):
        loss += t * log(p)

    return -1*loss