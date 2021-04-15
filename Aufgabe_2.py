import matplotlib.pyplot as plt
import math
from PIL import Image, ImageColor
import random


bias = 1
learn_rate = 0.01
num_patterns = 4

n = [3,5,1] # Abbildung des Neuronalen Netzes (Anzahl Neuronen pro Schicht)
bias_n = [0, 1, 0] # Kennzeichnung der Anzahl an Neuronen, die keine Kantengewichtungen haben
w = []
#w = [[[0.25, 0.34, 0.4], [-0.2, 0.12, 0.34], [0.3, 0.12, 0.000001], [-0.21, -0.12, 0.2]],[[-0.2, 0.1, -0.1, 0.24, 0.45]]] # Fixed weights for developement
x = [0, 0, 1, 1]
y = [0, 1, 0, 1] 
#t = [0, 0, 0, 1] # AND
#t = [0, 1, 1, 1] # OR
t = [0, 1, 1, 0] # XOR
hidden_output = [0,0,0,0] # Initiales Array für die Outputs der Hidden-Schicht
weight_mutiplier = 0.5


def init_weights(): # Generiert Initialgewichte für alle Neuronen in den Schichten 

    for i in range(len(n)-1): # Gewichte für n-1 Schichten generieren (da Gewichte nur zwischen den Schichten)

        outer_tmp = []

        for q in range(n[i+1] - bias_n[i+1]): # Gewichte für n Knoten der jeweiligen Schicht generieren

            inner_tmp = []

            for r in range(n[i]): # x Gewichte für jeden Knoten n der jeweiligen Schicht generieren

                inner_tmp.append(random.uniform(-1, 1) * weight_mutiplier)

            outer_tmp.append(inner_tmp)

        w.append(outer_tmp)


def get_neural_output_hidden(input, i): # input: x-Wert + y-Wert

    return 1/(1 + math.exp(-10*(bias * w[0][i][0] + input[0] * w[0][i][1] + input[1] * w[0][i][2])))


def get_neural_output_output(input, i): # input: Output Neuron 1 + Output Neuron 2 + Output Neuron 3 + Output Neuron 4

    return 1/(1 + math.exp(-10*(bias * w[1][i][0] + input[0] * w[1][i][1] + input[1] * w[1][i][2] + input[2] * w[1][i][3] + input[3] * w[1][i][4])))


def get_neural_output_final(x,y):

    #for i in range(n[1]-bias_n[1]):

    layer_1 = [get_neural_output_hidden([x, y], 0), get_neural_output_hidden([x, y], 1), get_neural_output_hidden([x, y], 2), get_neural_output_hidden([x, y], 3)]
    layer_2 = get_neural_output_output(layer_1, 0)

    return layer_2


def train():

    # Hidden-Layer trainieren

    for i in range(len(w[0])): # Für jedes Neuron der Schicht

        for q in range(num_patterns):

            input = [x[q], y[q]]

            out = get_neural_output_hidden(input, i)

            w[0][i][0] += learn_rate * (t[q]-out) * out * (1-out) * bias
            w[0][i][1] += learn_rate * (t[q]-out) * out * (1-out) * input[0]
            w[0][i][2] += learn_rate * (t[q]-out) * out * (1-out) * input[0]

            hidden_output[i] = get_neural_output_hidden(input, i)
    
    # Output-Layer trainieren

    for i in range(len(w[1])):

        for q in range(num_patterns):

            input = [hidden_output[0], hidden_output[1], hidden_output[2], hidden_output[3]]

            out = get_neural_output_output(input, i)

            w[1][i][0] += learn_rate * (t[q]-out) * out * (1-out) * bias
            w[1][i][1] += learn_rate * (t[q]-out) * out * (1-out) * input[0]
            w[1][i][2] += learn_rate * (t[q]-out) * out * (1-out) * input[1]
            w[1][i][3] += learn_rate * (t[q]-out) * out * (1-out) * input[2]
            w[1][i][4] += learn_rate * (t[q]-out) * out * (1-out) * input[3]
            
def main():

    init_weights()

    print("\nVor dem Trainieren:\n")
    print(f"Hidden-Layer:{w[0]}")
    print(f"Output-Layer:{w[1]}")
    print(f"Hidden Output: {hidden_output}")

    for i in range(1000000):

        train()

    print()
    print("\nNach dem Trainieren:\n")
    print(f"Hidden-Layer:{w[0]}")
    print(f"Output-Layer:{w[1]}")
    print(f"Hidden Output: {hidden_output}")

    im = Image.new('RGB', (500,500))

    x = 0
    while(x < 1):

        y = 0
        while(y < 1):

            out = get_neural_output_final(x, y)
            im.putpixel((int(x*500), int(499.98-y*500)), (int(out*255), int(out*255), int(out*255)))
            y += 0.002

        x += 0.002

    im.save('/Users/matteomuller/GNN/output3.png')
    

if __name__ == '__main__':

    main()



