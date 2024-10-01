from Unit import Unit
from ActivationFunctions import *
class FullyConnectedLayer:
    def __init__(self, width=8, activation='relu'):
        self.width=width
        self.create(width)
        self.set_activation(activation)

    def create(self, width):
        self.units = []
        for _ in range(width):
            self.units.append(Unit())

    def init_weights(self, size):
        for u in self.units:
            u.init_weights(size)

    def get_size(self):
        return self.width
    
    def get_units(self):
        return self.units
    
    def get_vals(self):
        #Get vector of activations
        vals = []
        for u in self.units:
            vals.append(u.value)
        return vals
    
    def set_vals(self, vals: list):
        for i, u in enumerate(self.units):
            u.value = vals[i]
    
    def set_activation(self, activation):
        act_funcs = {'relu' : relu,
                     'softmax' : softmax}
        self.activation_function = act_funcs.get(activation)

    def get_activation(self):
        return self.activation_function
    
    activation = property(get_activation)