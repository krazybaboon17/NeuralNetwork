from MLP import MLP
from autograd import Value
from Neuron import Neuron
from Layer import Layer
epochs = 2000
learningRate = 0.05
Input_Bits = 6
oddEven = MLP(Input_Bits, [3, 1])

data = [32, 5, 7, 9, 8, 10, 4, 3]
outputs = [1, 0, 0, 0, 1, 1, 1, 0]

for i in range(epochs):
    predY = []
    for x in data:
    
    #    predY.append(oddEven(x))
    #loss = 0
        #small to big
        bits = [Value((x >> bit) & 1) for bit in range(Input_Bits)] # - 1
        predY.append(oddEven(bits)[0])
        #predY.append(oddEven([Value(x)])[0]) #doesnt this return a list
    loss = Value(0.0) #maybe making it value works better

    for yActual, yPred in zip(outputs, predY):
        
        #loss += (yPred - yActual) ** 2
        #loss += (yPred + Value(-yActual)) ** 2 #i think this needs value too
        err = yPred + Value(-yActual)
        loss += err * err
        
    for parameter in oddEven.parameters():
        parameter.grad = 0.0
    loss.backward()

    for parameter in oddEven.parameters():
        parameter.data -= learningRate * parameter.grad
    
    if i % 100 == 0:
        print(f"Epoch {i} Loss: {loss.data}")
