import pandas as pd
import matplotlib.pyplot as plt


x = [7,-0.2,8]
delta_t = 0.01
liste = []


def calculate(x, delta_t):
    
    return x + delta_t * (x - x**3)


"""

Der Attraktor läuft gegen einen Fixpunkt:

Für positive Werte (7 und 8):
Wenn man für x 1 einsetzt, erhält man 1 + 0.01 * (1 - 1) => 1 + 0.01 * 0 => 1
-> Funktion geht gegen 1

Für negative Werte (-0.2):
Wenn man für x -1 einsetzt, erhält man -1 + 0.01 * (-1 -(-1)) => -1 + 0.01 * 0 => -1 
-> Funktion geht gegen -1


"""

# Pro Anfangsbedingung 1000 Berechnungen durchführen
for i in range(len(x)):

    x_new = x[i]

    for q in range(1000):
        
        x_new = calculate(x_new, delta_t)
        liste.append({
            'x': x_new,
            'y': q,
            'starting_condition': x[i]
        })
    
# Die berechneten Werte als DataFrame speichern, um sie daraufhin gut visualisieren zu können   
df_res = pd.DataFrame(liste)

# Alle Kurven einzeln visualisieren
for i in range(len(x)):

    plt.plot(df_res[df_res['starting_condition'] == x[i]]['y'], df_res[df_res['starting_condition'] == x[i]]['x'])

plt.title('GNN | Aufgabe 1')
plt.legend(x)
plt.show() 
