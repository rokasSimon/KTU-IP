import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

maxCap = 8192
maxSeq = 8000
maxTbw = 4000
maxCos = 1200


xSeq = np.arange(0, maxSeq, 1)
xTbw = np.arange(0, maxTbw, 1)
xCos = np.arange(0, maxCos, 1)

class Capacity:
    def __init__(self):
        self.xCap = np.arange(0, maxCap, 1)
        self.cap = [
            fuzz.trapmf(self.xCap, [0, 0, 120, 960]),
            fuzz.trapmf(self.xCap, [512, 960, 2048, 4096]),
            fuzz.trapmf(self.xCap, [2048, 4096, maxCap, maxCap]),
        ]

        return

    def score(self, c):

        scores = [
            fuzz.interp_membership(self.xCap, self.cap[0], c),
            fuzz.interp_membership(self.xCap, self.cap[1], c),
            fuzz.interp_membership(self.xCap, self.cap[2], c),
        ]

        return scores

    def plot(self):

        plt.figure()
        plt.plot(self.xCap, self.cap[0], 'b', label = "Maža")
        plt.plot(self.xCap, self.cap[1], 'g', label = "Vidutinė")
        plt.plot(self.xCap, self.cap[2], 'r', label = "Didelė")
        plt.title('SSD talpa (GB)')
        plt.legend()

        return

class Speed:
    def __init__(self):
        self.xSeq = np.arange(0, maxSeq, 1)
        self.seq = [
            fuzz.trapmf(self.xSeq, [0, 0, 520, 3100]),
            fuzz.trimf( self.xSeq, [1800, 3100, 4400]),
            fuzz.trapmf(self.xSeq, [3100, 4400, 6500, 7000]),
            fuzz.trapmf(self.xSeq, [6500, 7000, maxSeq, maxSeq]),
        ]

        return

    def score(self, s):

        scores = [
            fuzz.interp_membership(self.xSeq, self.seq[0], s),
            fuzz.interp_membership(self.xSeq, self.seq[1], s),
            fuzz.interp_membership(self.xSeq, self.seq[2], s),
            fuzz.interp_membership(self.xSeq, self.seq[3], s)
        ]

        return scores

    def plot(self):

        plt.figure()
        plt.plot(self.xSeq, self.seq[0], 'b', label = "Lėtas")
        plt.plot(self.xSeq, self.seq[1], 'g', label = "Vidutinis")
        plt.plot(self.xSeq, self.seq[2], 'r', label = "Greitas")
        plt.plot(self.xSeq, self.seq[3], 'y', label = "Labai greitas")
        plt.title('SSD nuoseklus skaitymo greitis (MB/s)')
        plt.legend()

        return

class TBW:
    def __init__(self):
        self.xTbw = np.arange(0, maxTbw, 1)
        self.tbw = [
            fuzz.trapmf(self.xTbw, [0, 0, 75, 600]),
            fuzz.trapmf(self.xTbw, [300, 600, 1200, 3200]),
            fuzz.trapmf(self.xTbw, [1200, 3200, maxSeq, maxSeq]),
        ]

        return

    def score(self, t):

        scores = [
            fuzz.interp_membership(self.xTbw, self.tbw[0], t),
            fuzz.interp_membership(self.xTbw, self.tbw[1], t),
            fuzz.interp_membership(self.xTbw, self.tbw[2], t),
        ]

        return scores

    def plot(self):

        plt.figure()
        plt.plot(self.xTbw, self.tbw[0], 'b', label = "Prasta")
        plt.plot(self.xTbw, self.tbw[1], 'g', label = "Vidutinė")
        plt.plot(self.xTbw, self.tbw[2], 'r', label = "Profesionaliam darbui")
        plt.title('SSD ištvermė (TBW)')
        plt.legend()

        return

class Cost:
    def __init__(self):
        self.xCos = np.arange(0, maxCos, 1)
        self.cost = [
            fuzz.trapmf(self.xCos, [0, 0, 20, 100]),
            fuzz.trapmf(self.xCos, [80, 100, 270, 380]),
            fuzz.trapmf(self.xCos, [270, 380, maxCos, maxCos]),
        ]

        return

    def plot(self):

        plt.figure()
        plt.plot(self.xCos, self.cost[0], 'b', label = "Biudžetinis")
        plt.plot(self.xCos, self.cost[1], 'g', label = "Mainstream")
        plt.plot(self.xCos, self.cost[2], 'r', label = "Entuziastams")
        plt.title('SSD vartotojų segmento kaina (€)')
        plt.legend()

        return


def getCostScore(cap, seq, tbw):

    lowCost, medCost, highCost = 0, 0, 0
    (lowCap, medCap, highCap) = Capacity().score(cap)
    (lowSeq, medSeq, highSeq, vhighSeq) = Speed().score(seq)
    (lowTbw, medTbw, highTbw) = TBW().score(tbw)
    (lowCostF, medCostF, highCostF) = Cost().cost



    return [
        np.fmin(lowCost, lowCostF),
        np.fmin(medCost, medCostF),
        np.fmin(highCost, highCostF)
    ]

def main():

    cap = Capacity()
    seq = Speed()
    tbw = TBW()
    cost = Cost()

    cap.plot()
    seq.plot()
    tbw.plot()
    cost.plot()

    plt.show()

    return

main()