import numpy as np

""""

output neuron --> activation function --> yPred --> loss

dL/dO = (1 - tanh^2(O))


"""
class Value:
    def __init__(self, data, children = ()):
        self.data = data
        self.children = children
        self.grad = 0
        self._backward = lambda: None


    def __add__(self, other):
        out = Value(other.data + self.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    def activation(self):
        out = Value(np.tanh(self.data), (self,))
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out
    def backward(self):
        topo = []
        visited = set()
        def buildTopo(v):
            if v not in visited:
                visited.add(v)
                for child in v.children:
                    buildTopo(child)
                topo.append(v)
        buildTopo(self)
        self.grad = 1
        for node in reversed(topo):
            node._backward()


