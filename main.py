from MLP import MLP
from autograd import Value
from Neuron import Neuron
from Layer import Layer
epochs = 2000
learningRate = 0.05
Input_Bits = 6

#oddEven = MLP(Input_Bits, [3, 1])
oddEven = MLP(1, [3, 1]) #changed to only get last digit
data = [32, 5, 7, 9, 8, 10, 4, 3]
outputs = [1, 0, 0, 0, 1, 1, 1, 0]

for i in range(epochs):
    predY = []
    for x in data:
    
    #    predY.append(oddEven(x))
    #loss = 0
        #small to big
        #bits = [Value((x >> bit) & 1) for bit in range(Input_Bits)] # - 1
        lastDigit = [Value(x % 10)]
        predY.append(oddEven(lastDigit)[0])
        #predY.append(oddEven([Value(x)])[0]) #doesnt this return a list
    loss = Value(0.0) #maybe making it value works better
    counter = 0
    for yActual, yPred in zip(outputs, predY):
        counter += 1
        #loss += (yPred - yActual) ** 2
        #loss += (yPred + Value(-yActual)) ** 2 #i think this needs value too
        err = yPred + Value(-yActual)
        loss += err * err
        loss = loss * Value(1 / counter) #average loss
        
    for parameter in oddEven.parameters():
        parameter.grad = 0.0
    loss.backward()

    for parameter in oddEven.parameters():
        parameter.data -= learningRate * parameter.grad
    
    if i % 100 == 0:
        print(f"Epoch {i} Loss: {loss.data}")
def test(x):
    bits = [Value((x >> bit) & 1) for bit in range(Input_Bits)] #doing ts convert to bits thing
    result = oddEven(bits)[0] # get ts result so we can show it
    print(result.data)

x = int(input("enter num: "))
test(x)
