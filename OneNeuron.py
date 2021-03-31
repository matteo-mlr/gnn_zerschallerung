import matplotlib.pyplot as plt
import math
from PIL import Image, ImageColor

bias = 1
num_patterns = 4
learn_rate = 2 

w = [0.3, -0.543, 0.122]
xx = [0, 0, 1, 1]
yy = [0, 1, 0, 1]
t = [0, 1, 1, 1] # OR
#t = [0, 0, 0, 1] # AND
#t = [0, 1, 1, 0] #XOR

def get_neural_output(x, y):

    return 1/(1 + math.exp(-10*(bias * w[0] + x * w[1] + y * w[2])))

def train():

    for i in range(num_patterns):

        out = get_neural_output(xx[i], yy[i])

        w[0] += learn_rate * (t[i]-out) * out * (1-out) * bias
        w[1] += learn_rate * (t[i]-out) * out * (1-out) * xx[i]
        w[2] += learn_rate * (t[i]-out) * out * (1-out) * yy[i]


def main():

    print(w)

    for i in range(100000):

        train()

    print(w)

    im = Image.new('RGB', (500,500))

    x = 0
    while(x < 1):

        y = 0
        while(y < 1):

            out = get_neural_output(x, y)
            im.putpixel((int(x*500), int(499.98-y*500)), (int(out*255), int(out*255), int(out*255)))
            y += 0.002

        x += 0.002

    im.save('/Users/matteomuller/Downloads/AndNeuron/output.png')

if __name__ == '__main__':

    main()


