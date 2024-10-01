from math import exp
def relu(input: list) -> list:
    return [max(0, i) for i in input]

def relu_derivative(input: list) -> list:
    return [1 if i>0 else 0 for i in input]

def softmax(input: list) -> list:
    out = []
    for i in range(len(input)):
        denom = 0.0
        for j in input:
            try:
                denom += exp(j)
            except OverflowError:
                denom = float('inf')
        out.append((exp(input[i])) / denom)
    return out