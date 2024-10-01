import numpy as np
class SGD:
    def __init__(self, instance, learning_rate=0.01, epochs=100):
        self.step_size = learning_rate
        self.instance = instance
        self.epochs = epochs

    def set_step_size(self, stp):
        self.step_size = stp
    def get_step_size(self):
        return self.step_size
    learning_rate = property(get_step_size, set_step_size)

    def get_gradients(self, X, y):
        y_pred = []
        for row_X in X:
            #find prediction on entire set
            y_pred.append(self.instance.predict(row_X))
        y_pred = np.array(y_pred)
        y = np.array(y)
        error = y_pred - y
        grad_w = np.dot(X.T, error) / X.shape[0]
        grad_b = np.mean(error)
        return grad_w, grad_b
    
    def train(self, X, y):
        for _ in range(self.epochs):
            grad_w, grad_b = self.get_gradients(X, y)
            weight_matrix = []
            for l in self.instance.get_weights():
                weight_matrix.append([])
                for u in l:
                    print(grad_w)
                    weight_matrix[-1].append(u - self.step_size * grad_w)

            self.instance.set_weights(weight_matrix)
            print('Accuracy: ' + str(self.instance.accuracy()))