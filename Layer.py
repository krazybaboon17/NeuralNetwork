from Neuron import Neuron
from autograd import Value

class Layer:
    def __init__(self, inputs, outputs):
        self.neurons = []
        for i in range(outputs):
            self.neurons.append(Neuron(inputs))
    def __call__(self, x):
        outs = []
        for neuron in self.neurons:
            outs.append(neuron(x))
        return outs
    def parameters(self):
        returnValue = []
        for neuron in self.neurons:
            for parameter in neuron.parameters():
                returnValue.append(parameter)
        return returnValue
    