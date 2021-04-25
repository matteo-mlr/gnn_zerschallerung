from math import exp
import random
import math
from PIL import Image, ImageColor


w_i = [
    [0.013, 0.034],
    [0.013, 0.024],
    [0.024, 0.064],
    [0.072, 0.014]
] # gewichte des input layer
w_h = [0.013, 0.034, 0.007, 0.012] # gewichte des hidden layer

# hier noch etwas ändern?
LEARNRATE = 1
EPOCHS = 100000

def init_weights():
    return []

def get_output_layer_out(x, y, sigmoid = True):

    # output = [h1,h2,h3,h4], output dient als input für hidden layer
    hidden_layer = get_hidden_layer_out(x, y) 
    
    # summe der neuronen des hidden layers richtung ausgabe layer (bias muss noch hinzugefügt werden)
    sum_weighted_output = hidden_layer[0] * w_h[0] + hidden_layer[1] * w_h[1] + hidden_layer[2] * w_h[2] + hidden_layer[3] * w_h[3]
    
    # wird hier schon die standardsigmoide gebraucht? oder erst später?
    return standard_sigmoid(sum_weighted_output) if sigmoid else sum_weighted_output

def get_hidden_layer_out(x,y):
    # out ist die Aktivität an den Neuronen des Hidden Layers
    out = [0,0,0,0]

    out[0] = x * w_i[0][0] + y * w_i[0][1]
    out[1] = x * w_i[1][0] + y * w_i[1][1]
    out[2] = x * w_i[2][0] + y * w_i[2][1]
    out[3] = x * w_i[3][0] + y * w_i[3][1]

    return out

def standard_sigmoid(arg:int, c:int = 10):
    return 1.0/(1.0 + exp(-c*(arg)))

def standard_sigmoid_derivation(arg:int):
    return standard_sigmoid(arg) * (1- standard_sigmoid(arg))

def train_one_epoch(x,y,t):

    o_j = get_hidden_layer_out(x, y) # Aktivität der Neuronen hidden layer
    o_k = get_output_layer_out(x, y) # Output der Neuronen ausgabe layer

    activity_o_k = get_output_layer_out(x, y, False) # sum weighted outputs of o_k

    e_k = (o_k - t) * standard_sigmoid_derivation(activity_o_k)
    
    #Backpropagation für ausgabe layer
    w_h[0] += -LEARNRATE * e_k * o_j[0]
    w_h[1] += -LEARNRATE * e_k * o_j[1]
    w_h[2] += -LEARNRATE * e_k * o_j[2]
    w_h[3] += -LEARNRATE * e_k * o_j[3]

    #Backpropagation für hidden layer
    #e_j wird pro Neuron im hidden layer gebildet

    #Neuron 1 Hidden Layer
    e_j = e_k * w_h[0] * standard_sigmoid_derivation(w_i[0][0] * standard_sigmoid(x) + w_i[0][1] * standard_sigmoid(y))

    w_i[0][0] += -LEARNRATE * e_j * standard_sigmoid(x)
    w_i[0][1] += -LEARNRATE * e_j * standard_sigmoid(y)

    #Neuron 2 Hidden Layer
    e_j = e_k * w_h[1] * standard_sigmoid_derivation(w_i[1][0] * standard_sigmoid(x) + w_i[1][1] * standard_sigmoid(y))

    w_i[1][0] += -LEARNRATE * e_j * standard_sigmoid(x)
    w_i[1][1] += -LEARNRATE * e_j * standard_sigmoid(y)

    #Neuron 3 Hidden Layer
    e_j = e_k * w_h[2] * standard_sigmoid_derivation(w_i[2][0] * standard_sigmoid(x) + w_i[2][1] * standard_sigmoid(y))

    w_i[2][0] += -LEARNRATE * e_j * standard_sigmoid(x)
    w_i[2][1] += -LEARNRATE * e_j * standard_sigmoid(y)

    #Neuron 4 Hidden Layer
    e_j = e_k * w_h[3] * standard_sigmoid_derivation(w_i[3][0] * standard_sigmoid(x) + w_i[3][1] * standard_sigmoid(y))

    w_i[3][0] += -LEARNRATE * e_j * standard_sigmoid(x)
    w_i[3][1] += -LEARNRATE * e_j * standard_sigmoid(y)

def is_in_unitcircle(x,y):
    x = abs(x)
    y = abs(y)
    return True if math.sqrt(x * x + y * y) <= 1 else False

def train():
    #init weights
  
    for i in range(EPOCHS):
        x = random.uniform(-1.5,1.5)
        y = random.uniform(-1.5,1.5)
        t = 1 if is_in_unitcircle(x, y) else 0

        train_one_epoch(x, y, t)    

def printImg():
    im = Image.new('RGB', (500, 500))

    x = -1
    while (x < 1):

        y = -1
        while (y < 1):
            out = get_output_layer_out(x, y)
            im.putpixel((int(x * 250 + 250), int(499.98 - (y * 250 + 250))), (int(out * 255), int(out * 255), int(out * 255)))
            y += 0.002

        x += 0.002

    im.save("./output.png")

def main():
    train()
    printImg()



main()
