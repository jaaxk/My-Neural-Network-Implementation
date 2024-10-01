import random
class RandomSearch:
    #Not recommended

    def __init__(self, instance, iterations = 1000):
        self.iterations = iterations
        self.instance = instance

    def set_iterations(self, itr):
        self.iterations = itr
    def get_iterations(self):
        return self.iterations

    iterations = property(fget = get_iterations, fset=set_iterations)

    def train(self, X, y):
        best_loss = float('inf')
        bestW = 0
        bestB = 0
        for _ in range(self.iterations):
            #Fill weights with random numbers
            weight_matrix = []
            for i, l in enumerate(self.instance.layers[0:-1]):
                weight_matrix.append([])
                for j, u in enumerate(l.get_units()):
                    weight_matrix[i].append([])
                    for w in u.weights:
                        weight_matrix[i][j].append(random.random())
            self.instance.set_weights(weight_matrix)

            #Find loss of entire dataset
            total_loss = 0.0
            for row_X, row_y in zip(X, y):
                total_loss += self.instance.loss_function(self.instance.predict(row_X), row_y)

            if total_loss < best_loss:
                best_loss = total_loss
                bestW = weight_matrix  
                print('Best loss (weights): ' + str(best_loss) + 'Accuracy: ' + str(self.instance.accuracy()))

        if bestW != 0:
            self.instance.set_weights(bestW)

        for _ in range(self.iterations):
            #Fill biases with random numbers
            bias_matrix = []
            for l in self.instance.layers:
                bias_matrix.append([])
                for u in l.units:
                    bias_matrix[-1].append(random.random())
            self.instance.set_biases(bias_matrix)
            
            #Find loss of entire dataset
            total_loss = 0.0
            for row_X, row_y in zip(X, y):
                total_loss += self.instance.loss_function(self.instance.predict(row_X), row_y)

            if total_loss < best_loss:
                best_loss = total_loss
                bestB = bias_matrix 
                print('Best loss (biases): ' + str(best_loss) + 'Accuracy: ' + str(self.instance.accuracy()))

        if bestB != 0:
            self.instance.set_biases(bestB)

        #print('Accuracy: ' + str(self.instance.accuracy()*100) + '%')
        return bestW, bestB