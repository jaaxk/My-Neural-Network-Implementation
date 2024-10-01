class Unit:
    def __init__(self):
        self._weights = []
        self._bias = 0
        self._value = 0.0
        #'Value' or 'Activation'

    def init_weights(self, size: int):
        self._weights = [1 for _ in range(size)]
        
    def calculate_value(self):
        pass

    def get_value(self):
        return self._value
    
    def set_value(self, value):
        self._value = value
    
    def get_weights(self):
        return self._weights
    
    def set_weights(self, weights: list):
        self._weights = weights
    
    def get_bias(self):
        return self._bias

    def set_bias(self, bias):
        self._bias = bias

    value = property(get_value, set_value)
    weights = property(get_weights, set_weights)
    bias = property(get_bias, set_bias)