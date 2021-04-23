import pygame, sys
from pygame.locals import *
import Aufgabe_2 as gnn

CANVAS_WIDTH = 300
CANVAS_HEIGHT = 300

RESOLUTION = 10

def main():

    #init canvas
    pygame.init()
    canvas = pygame.display.set_mode((CANVAS_WIDTH,CANVAS_HEIGHT))

    epoch_counter = 0
    gnn.init_weights()

    while True:

        #exit button
        for event in pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()

        #update x and y values

        if epoch_counter <= gnn.EPOCHS:
            epoch_counter += 1
            
            print(f"Epoch: {epoch_counter}")

            #training
            gnn.train()
            
            #drawing
            x = 0
            while x <= 1:
                y = 0
                while y <= 1:

                    #calculate color value for x and y with neural network
                    output = gnn.get_neural_output_final(x, y)
                    color = (output*255, output*255, output*255)
                    pos_x = x*CANVAS_WIDTH
                    pos_y = CANVAS_HEIGHT - RESOLUTION - y*CANVAS_HEIGHT #substract because we want (0,0) in bottom left corner

                    #draw pixels
                    pygame.draw.rect(canvas, color, (pos_x, pos_y, RESOLUTION, RESOLUTION))

                    y += 0.002 * RESOLUTION
                x += 0.002 * RESOLUTION

            #update canvas
            pygame.display.update()

main()