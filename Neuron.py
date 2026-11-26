import random
from autograd import Value
class Neuron:
    def __init__(self, inputs):
        self.w = []
        for i in range(inputs):
            self.w.append(Value(random.uniform(-1, 1)))
        self.b = Value(0.0)
    def __call__(self, x):
        out = 0
        for wi, xi in zip(self.w, x):
            out += wi * xi + self.b
        return out.tanh()
    def parameters(self):
        return self.w + list(self.b)
