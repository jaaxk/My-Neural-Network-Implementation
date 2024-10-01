import numpy as np
from Layers import FullyConnectedLayer
from Skull import Skull
from Preprocessing import to_categorical, normalize
import pandas as pd

X_train = np.array([[1,1,1],
         [4,2,4],
         [4,2,4],
         [1,1,1],
         [4,2,4]])

y_train = np.array([1,2,2,1,2])

a =  Skull()
a.add(FullyConnectedLayer(width=3))
a.add(FullyConnectedLayer())
a.add(FullyConnectedLayer(width=2, activation='softmax'))
a.train(X_train, to_categorical(y_train))


#print(a.predict([4,2,4]))

#X_test = [[1,1,1], [2,2,2], [4,2,4], [4,4,4], [5,6,7]]
#y_test = to_categorical([1, 2, 2, 1, 1])
#a.accuracy(X_test, y_test)
#print(a.predict([4,2,4]))