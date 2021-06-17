import math

"""

Gruppenmitglieder:

Knapp, Robin - 1823538
Delev, Daniel - 1821027
Müller, Matteo - 1824001

"""

# Führt zur Berechnung der nächsten vier Werte
N_ITERATIONS = 2

# Initialgewicht und Input
Wbias1 = -3.37
Wbias2 = 0.125
W11 = -4
W12 = 1.5
W21 = -1.5
W22 = 0
o1 = 0
o2 = 0
transfer_weight = 1  # Variable von uns ergänzt um Rückkopplungsgewicht darzustellen

if __name__ == '__main__':
    i = 0
    print(f'initial inputs: {o1} / {o2}')
    while i < N_ITERATIONS:
        # Vorwärtsaktivierung
        active_1 = o1 * W11 + o2 * W21 + 1 * Wbias1
        active_2 = o1 * W12 + o2 * W22 + 1 * Wbias2
        # Transfersfunktion mittels Tangens-Hyperbolicus
        out1 = math.tanh(active_1)
        out2 = math.tanh(active_2)
        # aktueller Output dient als Input für nächste Vorwärtsaktivierung
        o1 = out1 * transfer_weight  # Aktivierung "zwischen" den Zuständen bzw. vor der Rückkopplung
        o2 = out2 * transfer_weight
        print(f'[{i + 1}] {o1} / {o2}')
        i += 1
