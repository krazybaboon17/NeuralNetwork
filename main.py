from MLP import MLP
from autograd import Value
from Neuron import Neuron
from Layer import Layer
epochs = 2000
learningRate = 0.05
oddEven = MLP(1, [3, 1])

data = [32, 5, 7, 9, 8, 10, 4, 3]
outputs = [1, 0, 0, 0, 1, 1, 1, 0]

for i in range(epochs):
    predY = []
    for x in data:
        predY.append(oddEven(x))
    loss = 0
    for yActual, yPred in zip(outputs, predY):
        loss += (yPred - yActual) ** 2
    for parameter in oddEven.parameters():
        parameter.grad = 0.0
    loss.backward()

    for parameter in oddEven.parameters():
        parameter.data -= learningRate * parameter.grad
    
    if i % 100 == 0:
        print(f"Epoch {i} Loss: {loss.data}")



    

