from cProfile import label
from matplotlib import projections
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def readSunspotActivity(filePath: str) -> tuple[list[int], list[int]]:
    years = []
    activity = []

    with open(filePath, 'r') as file:
        for line in file:
            [year, act] = line.split('\t')
            years.append(int(year))
            activity.append(int(act))

    return (years, activity)

def plotSunspotActivity(years: list[int], activity: list[int]):

    plt.figure()
    plt.plot(years, activity)
    plt.title("Sun activity 1700-2014")
    plt.xlabel("Years")
    plt.ylabel("Sunspot number")

    return

def prepareInputs(activity: list[int], n: int) -> tuple[list[int], list[int]]:

    p, t = [], []
    count = len(activity)

    for i in range(count - n):
        p.append(activity[i:i+n])

    t = activity[n:]

    return (p, t)

def plotSunspot3D(inputs: list[list[int]], outputs: list[int]):

    x = [act[0] for act in inputs]
    y = [act[1] for act in inputs]
    z = outputs

    plt.figure()
    ax = plt.axes(projection='3d')

    ax.scatter3D(x, y, z, 'blue')

    ax.set_xlabel('First input')
    ax.set_ylabel('Second input')
    ax.set_zlabel('Output')
    ax.set_title('Input and output 3D representation')

    return

def plotPredictionsToReal(years: list[int], real: list[int], predictions: list[float]):

    plt.figure()
    plt.plot(years, real, 'blue', label='Test data')
    plt.plot(years, predictions, 'red', label='Prediction data')
    plt.title('Comparison of real data and prediction')
    plt.legend()

    return

def plotDifference(years: list[int], error: list[float]):

    plt.figure()
    plt.xlabel('Years')
    plt.ylabel('Difference')
    plt.plot(years, error, 'blue')
    plt.title('Prediction error graph')

    return

def plotDifferenceHist(error: list[float]):

    plt.figure()
    plt.hist(error)
    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.title('Prediction error histogram')

    return

def main():

    # 1, 2
    (years, activity) = readSunspotActivity('sunspot.txt')

    # 3
    if [years[0], activity[0]] != [1700, 5]:
        print("First sunspot listing doesn't match")
        return
    
    # 4
    #plotSunspotActivity(years, activity)

    # 5
    degree = 2
    [P, T] = prepareInputs(activity, degree)
    print(f"Input list size: {len(P)}")
    print(f"Output list size: {len(T)}")

    # 6
    #plotSunspot3D(P, T)

    # 7
    lim = 200
    [Pu, Tu] = [np.array(P[:lim]).reshape(-1, degree), np.array(T[:lim]).reshape(-1, 1)]                       # training data
    [testPu, testTu] = [np.array(P[lim-degree:]).reshape(-1, degree), np.array(T[lim-degree:]).reshape(-1, 1)] # testing data

    # 8
    model = LinearRegression().fit(Pu, Tu)

    # 9
    [w1, w2] = model.coef_[0]
    b = model.intercept_[0]
    print(f"Coefficients: [{w1} {w2}]")
    print(f"Intercept: {b}")

    # 10
    Tsu = model.predict(Pu)
    #plotPredictionsToReal(years[:lim], Tu, Tsu)

    testTsu = model.predict(testPu)
    #plotPredictionsToReal(years[lim:], testTu, testTsu)

    # 11
    e = testTsu - testTu
    #plotDifference(years[lim:], e)

    # 12
    plotDifferenceHist(e)

    # 13

    plt.show()
    return

main()