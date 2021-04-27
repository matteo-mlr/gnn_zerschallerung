import math
import random
from math import exp

from PIL import Image

# gewichte input-layer --> hidden-layer, letzte jeweils für bias
w_i = [
    [0.013, 0.034, 0.09],
    [0.013, 0.024, 0.0321],
    [0.024, 0.064, 0.098],
    [0.072, 0.014, 0.0918]
]

# gewichte hidden-layer --> output-layer, letzte für bias
w_h = [0.013, 0.034, 0.007, 0.012, 0.09]

# hier noch etwas ändern?
LEARNRATE = 1
EPOCHS = 20
BIAS = 1


def init_weights():
    pass


def get_output_layer_activity(hidden_layer_output: list[float]) -> float:
    """ hidden_layer_output: Enthält bereits sigmoide Werte"""
    sum_weighted_output = hidden_layer_output[0] * w_h[0] + \
                          hidden_layer_output[1] * w_h[1] + \
                          hidden_layer_output[2] * w_h[2] + \
                          hidden_layer_output[3] * w_h[3] + \
                          BIAS * w_h[4]
    return sum_weighted_output


def get_hidden_layer_activity(x: float, y: float) -> list[float]:
    sum_weighted_output = [0, 0, 0, 0]

    sum_weighted_output[0] = x * w_i[0][0] + y * w_i[0][1] + BIAS * w_i[0][2]
    sum_weighted_output[1] = x * w_i[1][0] + y * w_i[1][1] + BIAS * w_i[1][2]
    sum_weighted_output[2] = x * w_i[2][0] + y * w_i[2][1] + BIAS * w_i[2][2]
    sum_weighted_output[3] = x * w_i[3][0] + y * w_i[3][1] + BIAS * w_i[3][2]

    return sum_weighted_output


def standard_sigmoid(activity: float, c: int = 10) -> float:
    """ activity: Aktivitaet am neuron (Summe der gewichteten Outputs)"""
    return 1.0 / (1.0 + exp(-c * activity))


def standard_sigmoid_derivation(sigmoidish_output: float) -> float:
    """ sigmoidish_output: sigmoide(aktivitaet)"""
    return sigmoidish_output * (1 - sigmoidish_output)


def predict(x: float, y: float):
    """ Forwardpropagation fürs Zeichnen der Punkte """
    activity_o_j = get_hidden_layer_activity(x, y)
    output_o_j = [standard_sigmoid(act) for act in activity_o_j]
    activity_o_k = get_output_layer_activity(output_o_j)
    return standard_sigmoid(activity_o_k)


def train_one_epoch(x: float, y: float, t: int):
    # Forwardpropagation um Ausgabe zu erhalten
    activity_o_j = get_hidden_layer_activity(x, y)  # Aktivität der NEURONEN hidden layer
    o_j = [standard_sigmoid(act) for act in activity_o_j]  # Output der NEURONEN hidden layer

    activity_o_k = get_output_layer_activity(o_j)  # Aktivität des output layer NEURON
    o_k = standard_sigmoid(activity_o_k)  # Output des output layer NEURON

    # Backpropagation für output layer
    error_k = (t - o_k) * standard_sigmoid_derivation(sigmoidish_output=o_k)  # Fehlergradient der Ausgabeschicht

    print(f'squared_error: {(t - o_k) ** 2}')

    # Backpropagation für hidden layer
    errors_j = []
    for i in range(len(w_h) - 1):  # für alle außer bias
        errors_j.append(error_k * w_h[i] * standard_sigmoid_derivation(sigmoidish_output=o_j[i]))

    update_weights_h(o_j, error_k)
    update_weights_i(o_k, errors_j)


def update_weights_h(o_j: list[float], error_k: float):
    """ o_j: sigmoide outputs"""
    w_h[0] += -LEARNRATE * error_k * o_j[0]
    w_h[1] += -LEARNRATE * error_k * o_j[1]
    w_h[2] += -LEARNRATE * error_k * o_j[2]
    w_h[3] += -LEARNRATE * error_k * o_j[3]
    w_h[4] += -LEARNRATE * error_k * BIAS


def update_weights_i(o_k: float, errors_j: list[float]):
    """ o_k: sigmoide outputs"""
    for i in range(len(w_h) - 1):  # für den hidden layer muss man auf die w_h verbindungen schauen
        w_i[i][0] += -LEARNRATE * errors_j[i] * o_k
        w_i[i][1] += -LEARNRATE * errors_j[i] * o_k
        w_i[i][2] += -LEARNRATE * errors_j[i] * BIAS


def is_in_unitcircle(x: float, y: float) -> bool:
    x = abs(x)
    y = abs(y)
    return True if math.sqrt(x * x + y * y) <= 1 else False


def train():
    # init weights

    for i in range(EPOCHS):
        x = random.uniform(-1.5, 1.5)
        y = random.uniform(-1.5, 1.5)
        t = 1 if is_in_unitcircle(x, y) else 0

        train_one_epoch(x, y, t)


def printImg():
    im = Image.new('RGB', (500, 500))

    x = -1
    while (x < 1):

        y = -1
        while (y < 1):
            out = predict(x, y)
            im.putpixel((int(x * 250 + 250), int(499.98 - (y * 250 + 250))),
                        (int(out * 255), int(out * 255), int(out * 255)))
            y += 0.002

        x += 0.002

    im.save("./output.png")


def main():
    random.seed(1)
    train()
    # printImg()


main()
