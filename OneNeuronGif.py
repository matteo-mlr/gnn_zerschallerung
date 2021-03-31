import matplotlib.pyplot as plt
import math
from PIL import Image, ImageColor
import imageio

bias = 1
num_patterns = 4
learn_rate = 2 

w = [0.3, -0.543, 0.122]
xx = [0, 0, 1, 1]
yy = [0, 1, 0, 1]
#t = [0, 1, 1, 1] # OR
t = [0, 0, 0, 1] # AND
#t = [0, 1, 1, 0] #XOR

def get_neural_output(x, y):

    return 1/(1 + math.exp(-10*(bias * w[0] + x * w[1] + y * w[2])))

def train():

    for i in range(num_patterns):

        out = get_neural_output(xx[i], yy[i])

        w[0] += learn_rate * (t[i]-out) * out * (1-out) * bias
        w[1] += learn_rate * (t[i]-out) * out * (1-out) * xx[i]
        w[2] += learn_rate * (t[i]-out) * out * (1-out) * yy[i]


def draw_image(i):

    im = Image.new('RGB', (250,250))

    x = 0
    while(x < 1):

        y = 0
        while(y < 1):

            out = get_neural_output(x, y)
            im.putpixel((int(x*250), int(249.98-y*250)), (int(out*255), int(out*255), int(out*255)))
            y += 0.004

        x += 0.004

    return im
    #im.save('./GNN/output_' + str(i) + '.png')


def main():

    epochs = 10000
    images = []

    for i in range(epochs):

        train()
        
        if i % 50 == 0:

            images.append(draw_image(i))
            print(str(i/epochs*100) + "%")

    imageio.mimsave('./GNN/output.gif', images)


if __name__ == '__main__':

    main()


