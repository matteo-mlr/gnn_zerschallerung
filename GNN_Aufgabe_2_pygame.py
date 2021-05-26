import pygame, sys
from pygame.locals import *
import Aufgabe2_Final as gnn


"""

Gruppenmitglieder:

Knapp, Robin - 1823538
Delev, Daniel - 1821027
Müller, Matteo - 1824001

"""

CANVAS_WIDTH = 200
CANVAS_HEIGHT = 200

RESOLUTION = 10

EPOCHS = 1000
N_TESTDATA = 1000

def main():

    #init canvas
    pygame.init()
    canvas = pygame.display.set_mode((CANVAS_WIDTH*2,CANVAS_HEIGHT*2))

    epoch_counter = 0
    gnn.init_network()
    testdata = gnn.generate_testdata(N_TESTDATA)

    while True:

        #exit button
        for event in pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()

        if epoch_counter < EPOCHS:
            epoch_counter += 1
            
            print(f"Epoch: {epoch_counter}")

            #training
            gnn.train_one_epoch_for_drawing(testdata)
            
            #drawing
            x = -1
            while x <= 1:
                y = -1
                while y <= 1:

                    #farbwerte für pixel an stelle (x,y) bestimmen mit neuralem netz
                    output = gnn.get_network_output([x,y])[1][0]
                    color = (output*255, output*255, output*255)
                    pos_x = x*CANVAS_WIDTH + CANVAS_WIDTH
                    pos_y = CANVAS_HEIGHT - RESOLUTION - y*CANVAS_HEIGHT # versatz damit (0,0) in mitte des canvas ist

                    #draw pixel
                    pygame.draw.rect(canvas, color, (pos_x, pos_y, RESOLUTION, RESOLUTION))

                    y += 0.002 * RESOLUTION
                x += 0.002 * RESOLUTION

            #update canvas
            pygame.display.update()

if __name__ == "__main__":
    main()