import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

maxCap = 8192
maxSeq = 8000
maxTbw = 4000
maxCos = 1200

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

def AND(*args):
    return np.min(args)

def OR(*args):
    return np.max(args)

def NOT(val):
    return 1 - val;

def lowCostRule(cap, seq, tbw):
    return OR(
        AND(cap[0], NOT(seq[3]), tbw[0]),
        AND(cap[1], seq[0], tbw[0]),
    )

def medCostRule(cap, seq, tbw):
    return OR(
        AND(cap[0], NOT(seq[3]), tbw[1]),
        AND(cap[0], seq[3], tbw[0]),
        AND(cap[1], seq[0], tbw[1]),
        AND(cap[1], OR(seq[1], seq[2]), OR(tbw[0], tbw[1])),
        AND(cap[2], NOT(seq[3]), tbw[0]),
        AND(cap[2], OR(seq[0], seq[1]), tbw[1]),
    )

def highCostRule(cap, seq, tbw):
    return OR(
        AND(tbw[2]),
        AND(seq[3], tbw[1]),
        AND(OR(cap[1], cap[2]), seq[3], tbw[0]),
        AND(cap[2], seq[2], tbw[1]),
    )

def implication(cap, seq, tbw, plot):

    c = Cost()
    y0 = np.zeros_like(c.xCos)

    capScore = Capacity().score(cap)
    seqScore = Speed().score(seq)
    tbwScore = TBW().score(tbw)
    (lowCostF, medCostF, highCostF) = c.cost

    lowCost  =  lowCostRule(capScore, seqScore, tbwScore)
    medCost  =  medCostRule(capScore, seqScore, tbwScore)
    highCost = highCostRule(capScore, seqScore, tbwScore)

    if plot:
        print(f"    Talpos įvertinimas: {capScore}")
        print(f"    Greičio įvertinimas: {seqScore}")
        print(f"    Tvermės įvertinimas: {tbwScore}")
        print(f"    Kainos įvertis: {[lowCost, medCost, highCost]}")

    lowImpl = np.fmin(lowCost, lowCostF)
    medImpl = np.fmin(medCost, medCostF)
    highImpl = np.fmin(highCost, highCostF)

    if plot:
        plt.figure()
        plt.plot(c.xCos, c.cost[0], 'green', label = "Biudžetinis")
        plt.fill_between(c.xCos, y0, lowImpl, facecolor = 'r', alpha = 0.5)
        plt.plot(c.xCos, c.cost[1], 'green', label = "Mainstream")
        plt.fill_between(c.xCos, y0, medImpl, facecolor = 'g', alpha = 0.5)
        plt.plot(c.xCos, c.cost[2], 'green', label = "Entuziastam")
        plt.fill_between(c.xCos, y0, highImpl, facecolor = 'b', alpha = 0.5)
        plt.title('Implikacija')

    return [lowImpl, medImpl, highImpl]

def aggregation(cap, seq, tbw, plot):

    (lowCost, medCost, highCost) = implication(cap, seq, tbw, plot)

    aggregated = np.fmax(
       lowCost,
       np.fmax(
           medCost, highCost
       )
    )

    return aggregated

def defuzzification(cap, seq, tbw, method, plot):

    c = Cost()

    agg = aggregation(cap, seq, tbw, plot)

    if plot:
        plt.figure()
        plt.plot(c.xCos, c.cost[0], 'black', zorder = 1)
        plt.plot(c.xCos, c.cost[1], 'black', zorder = 1)
        plt.plot(c.xCos, c.cost[2], 'black', zorder = 1)
        plt.plot(c.xCos, agg, 'blue', zorder = 2)
        plt.fill_between(c.xCos, np.zeros_like(c.xCos), agg, facecolor = 'blue')
        plt.title('Agregacija')

    return fuzz.defuzz(Cost().xCos, agg, method)

def main():

    cap = Capacity()
    seq = Speed()
    tbw = TBW()
    cost = Cost()

    cap.plot()
    seq.plot()
    tbw.plot()
    cost.plot()

    cases = [
        (256, 2000, 250),
        (2500, 3500, 1400),
        (6000, 7000, 3600)
    ]

    i = 0
    for case in cases:
        i += 1

        print(f"Scenarijus {i}:")
        print(f"    Talpa: {case[0]} GB")
        print(f"    Greitis: {case[1]} MB\\s")
        print(f"    Tvermė: {case[2]} TBW")

        resCentroid = defuzzification(*case, method='centroid', plot = True)
        resMom = defuzzification(*case, method='mom', plot = False)

        print(f"    Rezultatas (centroido): {resCentroid}")
        print(f"    Rezultatas (maksimumų vidurkis): {resMom}")
        print()

    plt.show()

    return

main()