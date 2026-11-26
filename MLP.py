from autograd import Value
from Layer import Layer
from Neuron import Neuron

class MLP:
    def __init__(self, inputs, sizes):
        sizes = [inputs] + sizes
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1]))
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        returnValue = []
        for layer in self.layers:
            for parameter in layer.parameters():
                returnValue.append(parameter)
        return returnValue

        
