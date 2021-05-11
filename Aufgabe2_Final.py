import math
import random
from PIL import Image, ImageColor
from random import uniform


network = []
output = [[[0],[0],[0],[0]],[[0]]]
delta_weights = [[[0],[0],[0],[0]],[[0]]]

LEARNRATE = 0.1

def init_network():
    #hidden layer, last weight = bias
    network.append([
        [0.3897357306683825, 0.9126469444394134, 0.7758730482768014], # Neuron 1, a neuron is essentially a list of weights
        [0.1735232013605318, 0.5978860651946428, 0.18040484098694443], # Neuron 2
        [0.7757205369141853, 0.5564327542525381, 0.7985897933821094], # Neuron 3
        [0.06492748093164047, 0.9280062358832749, 0.22981106076880609], # Neuron 4
    ])
    #output layer, last weight = bias
    network.append([
        [0.8497813814803977, 0.4415974940009172, 0.8888504188216229, 0.10164184412603228, 0.053793242180075196], # Neuron 1
    ])

def standard_sigmoid(arg:int, c:int = 1):
    return 1.0/(1.0 + math.exp(-c*arg))

def standard_sigmoid_derivation(arg:int):
    return arg * (1- arg)

# iterate over every weight of a neuron and multiply with the given inputs, last weight is always a bias with factor 1
def get_neuron_activation(neuron, inputs):
    sum = 0
    for i in range(len(neuron)-1):
        sum += inputs[i] * neuron[i]

    sum += 1 * neuron[-1] # bias 
    return sum

# inputs bsp: [0.3, 0.5]
def get_network_output(inputs):

    for i, layer in enumerate(network):
        inputs_new = []
        for j, neuron in enumerate(layer):
            output[i][j] = standard_sigmoid(get_neuron_activation(neuron, inputs))
            inputs_new.append(output[i][j])
        inputs = inputs_new

    return output

def backward_propagation(target):

    # für jedes neuron im output layer (1x mal)
    for j in range(len(network[1])):

        error = target - output[1][j] # fehler berrechnen
        delta_weights[1][j] = error * standard_sigmoid_derivation(output[1][j]) # gewichtsveränderung berechnen und speichern


    # für jedes neuron im hidden layer (4x mal)
    for j in range(len(network[0])): 
        error = 0
        for neuron in network[1]: # für jedes output neuron (1x mal)
            error += neuron[j] * delta_weights[1][0] # fehler berechnen (= Fehler am Output Neuron * Gewicht)
        
        delta_weights[0][j] = error * standard_sigmoid_derivation(output[0][j]) # gewichtsveränderung berechnen und speichern
    
def update_weights(inputs):

    # hidden layer
    for i, neuron in enumerate(network[0]): # i zum iterieren über alle delta_weights
        for j in range(len(inputs)):
            neuron[j] += LEARNRATE * delta_weights[0][i] * inputs[j]
        neuron[-1] += LEARNRATE * delta_weights[0][i] * 1 # bias

    # output layer
    for neuron in network[1]:
        for j in range(len(output[0])): # inputs für den output layer sind die outputs des hidden layer
            neuron[j] += LEARNRATE * delta_weights[1][0] * output[0][j]
        neuron[-1] += LEARNRATE * delta_weights[1][0] * 1 # bias        

def is_in_unitcircle(x,y):
    x = abs(x)
    y = abs(y)
    return 1 if math.sqrt(x * x + y * y) <= 1 else 0

def printImg():
    im = Image.new('RGB', (500, 500))

    x = -1
    while (x < 1):

        y = -1
        while (y < 1):
            out = get_network_output([x,y])[1][0]

            im.putpixel((int(x * 250 + 250), int(499.98 - (y * 250 + 250))), (int(out * 255), int(out * 255), int(out * 255)))
            y += 0.002

        x += 0.002
    im.save("./KREIS.png")

def generate_testdata(n):
    testdata = []
    for i in range(n):
        x = uniform(-1.5,1.5)
        y = uniform(-1.5,1.5)
        t = is_in_unitcircle(x, y)
        testdata.append(((x, y), t))
    return testdata

def train(testdata, epochs):
    
    for i in range(epochs):
        sum_error = 0
        for data in testdata:
            inputs = data[0]
            target = data[1]

            get_network_output(inputs)
            sum_error += (target - output[1][0])**2
            backward_propagation(target)
            update_weights(inputs)

        print(f"Epoch: {i}; Error: {sum_error}")


def main():
    init_network()
    testdata = generate_testdata(1000)
    train(testdata, 1000)
    printImg()


main()