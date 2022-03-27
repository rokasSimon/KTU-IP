import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

maxCap = 8192
maxSeq = 8000
maxTbw = 4000
maxCos = 1200

xCap = np.arange(0, maxCap, 1)
xSeq = np.arange(0, maxSeq, 1)
xTbw = np.arange(0, maxTbw, 1)
xCos = np.arange(0, maxCos, 1)

cap = [
    fuzz.trapmf(xCap, [0, 0, 120, 960]),
    fuzz.trapmf(xCap, [512, 960, 2048, 4096]),
    fuzz.trapmf(xCap, [2048, 4096, maxCap, maxCap]),
]

seq = [
    fuzz.trapmf(xSeq, [0, 0, 520, 3100]),
    fuzz.trimf( xSeq, [1800, 3100, 4400]),
    fuzz.trapmf(xSeq, [3100, 4400, 6500, 7000]),
    fuzz.trapmf(xSeq, [6500, 7000, maxSeq, maxSeq]),
]

tbw = [
    fuzz.trapmf(xTbw, [0, 0, 75, 600]),
    fuzz.trapmf(xTbw, [300, 600, 1200, 3200]),
    fuzz.trapmf(xTbw, [1200, 3200, maxSeq, maxSeq]),
]

cost = [
    fuzz.trapmf(xCos, [0, 0, 20, 100]),
    fuzz.trapmf(xCos, [80, 100, 270, 380]),
    fuzz.trapmf(xCos, [270, 380, maxCos, maxCos]),
]

def main():

    #fig, ax = plt.subplots(nrows = 4)

    plt.figure()
    plt.plot(xCap, cap[0], 'b', label = "Maža")
    plt.plot(xCap, cap[1], 'g', label = "Vidutinė")
    plt.plot(xCap, cap[2], 'r', label = "Didelė")
    plt.title('SSD talpa (GB)')
    plt.legend()

    plt.figure()
    plt.plot(xSeq, seq[0], 'b', label = "Lėtas")
    plt.plot(xSeq, seq[1], 'g', label = "Vidutinis")
    plt.plot(xSeq, seq[2], 'r', label = "Greitas")
    plt.plot(xSeq, seq[3], 'y', label = "Labai greitas")
    plt.title('SSD nuoseklus skaitymo greitis (MB/s)')
    plt.legend()

    plt.figure()
    plt.plot(xTbw, tbw[0], 'b', label = "Prasta")
    plt.plot(xTbw, tbw[1], 'g', label = "Vidutinė")
    plt.plot(xTbw, tbw[2], 'r', label = "Profesionaliam darbui")
    plt.title('SSD ištvermė (TBW)')
    plt.legend()
    
    plt.figure()
    plt.plot(xCos, cost[0], 'b', label = "Biudžetinis")
    plt.plot(xCos, cost[1], 'g', label = "Mainstream")
    plt.plot(xCos, cost[2], 'r', label = "Entuziastams")
    plt.title('SSD vartotojų segmento kaina (€)')
    plt.legend()

    plt.show()

    return

main()