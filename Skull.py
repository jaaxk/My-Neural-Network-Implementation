#Structure that holds neural layers - Skull
import numpy as np
from ActivationFunctions import *
from OptimizationFunctions.RandomSearch import RandomSearch
from OptimizationFunctions.SGD import SGD
from LossFunctions import *

class Skull:
    def __init__(self, initial_layers: list = []):
        if not isinstance(initial_layers, list):
            raise TypeError('Input must be list of layer objects')
        self.layers = []
        for l in initial_layers:
            self.add(l)
        self.optimization_function = SGD(self)
        self.loss_function = cross_entropy

    def add(self, layer):
        self.layers.append(layer)
        if len(self.layers) == 1:
            return
        else:
            self.layers[-2].init_weights(layer.get_size())

    def predict(self, values: list):
        if len(values) != self.layers[0].get_size():
            raise IndexError('Input must be same size as first layer')
        #Set first layer equal to input
        self.layers[0].set_vals(self.layers[0].activation(values))
        #Iterate through layers
        for i, l in enumerate(self.layers[1::]):
            vals = []
            #Iterate through units
            for j, u_current in enumerate(l.get_units()):
                val = 0.0
                #Iterate through all units of PREVIOUS layer
                for u_prev in self.layers[i].get_units():
                    #Get value*weight of each previous unit
                    val += u_prev.value*u_prev.weights[j]
                    
                #Add bias of current unit
                val += u_current.bias
                
                vals.append(val)
            #Pass through activation function
            l.set_vals(l.activation(vals))

        return self.layers[-1].get_vals()

    def describe(self) -> str:
        if len(self.layers) != 0:
            description = 'Input: '
            for i, l in enumerate(self.layers):
                i += 1
                if i == len(self.layers):
                    description += 'Output: '
                description += 'Layer: '+ str(i) + ' Units: ' + str(l.get_size()) + '\n'
        print(description)
        return description
    
    def compile(self, optimization: str, loss: str, **kwargs):
        loss_funcs = {'cross_entropy': cross_entropy}
        opt_funcs = {'random': RandomSearch(self)}
        self.optimization_function = opt_funcs.get(optimization)
        for key, value in kwargs.items():
            self.optimization_function.key = value
        self.loss_function = loss_funcs.get(loss)


    def train(self, X, y):
        self.X_train =  X
        self.y_train = y
        self.optimization_function.train(X, y)
        
    
   

    #Find loss over entire dataset    
    def find_loss(self, X, y):
        loss = 0.0
        for row in X:
            pred = self.predict(row)
            loss += self.loss_function(pred, y)
        return loss
    
    def get_weights(self):
        weight_matrix = []
        for l in self.layers[0:-1]:
            weight_matrix.append([])
            for u in l.get_units():
                weight_matrix[-1].append(u.weights)
        return weight_matrix
    
    def set_weights(self, weight_matrix):
        """ Takes in list of lists of weights of each unit.
          Same size as return value of Skull.get_weights()"""
        for i, l in enumerate(self.layers[0:-1]):
            for j, u in enumerate(l.get_units()):
                u.weights = weight_matrix[i][j]

    def get_biases(self):
        bias_matrix = []
        for l in self.layers:
            bias_matrix.append([])
            for u in l.get_units():
                bias_matrix[-1].append(u.bias)
        return bias_matrix
    
    def set_biases(self, bias_matrix):
        for i, l in enumerate(self.layers):
            for j, u in enumerate(l.units):
                u.bias = bias_matrix[i][j]

    def accuracy(self, X_test = None, y_test = None):
        """Tests accuracy of model using testing set, or, by default, previously given training data
            Checks if argmax of predicted value is same as true value"""
        if (X_test is None) & (y_test is None):
            X_test = self.X_train
            y_test = self.y_train

        correct = 0
        for X, y in zip(X_test, y_test):
            if np.argmax(self.predict(X)) == np.argmax(y):
                correct += 1
        return correct / len(X_test)
        




