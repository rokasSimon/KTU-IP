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
    plt.title("Sunspot activity 1700-2014")
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

def plotPredictionsToReal(years: list[int], real: list[int], predictions: list[float], title: str):

    plt.figure()
    plt.plot(years, real, 'blue', label='Test data')
    plt.plot(years, predictions, 'red', label='Prediction data')
    plt.title(title)
    plt.legend()
    plt.xlabel('Years')
    plt.ylabel('Sunspot value')

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

def MSE(e: list[float]) -> float:

    sqr = np.square(e)

    return 1 / len(e) * sqr.sum()

def MAD(e: list[float]) -> float:

    return np.median(np.abs(e))

def normalize(x):
    xnd = np.asarray(x)
    return (xnd - xnd.min()) / (np.ptp(x))

class AdaptiveLinearNeuron(object):
   def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niter = niter

   def fit(self, X, y):
      """Fit training data
      X : Training vectors, X.shape : [#samples, #features]
      y : Target values, y.shape : [#samples]
      """

      # weights
      self.weight = np.zeros(1 + X.shape[1])

      # Number of misclassifications
      self.errors = []

      # Cost function
      self.cost = []

      for i in range(self.niter):
         output = self.net_input(X)
         errors = y - output
         self.weight[1:] += self.rate * X.T.dot(errors)
         self.weight[0] += self.rate * errors.sum()
         cost = (errors**2).sum() / 2.0

         self.cost.append(cost)
         self.errors.append(errors)
      return self

   def net_input(self, X):
      """Calculate net input"""
      return np.dot(X, self.weight[1:]) + self.weight[0]

   def activation(self, X):
      """Compute linear activation"""
      return self.net_input(X)

   def predict(self, X):
      """Return class label after unit step"""
      return np.where(self.activation(X) >= 0.0, 1, -1)

def main():

    # 1, 2
    (years, activity) = readSunspotActivity('sunspot.txt')

    # 3
    if [years[0], activity[0]] != [1700, 5]:
        print("First sunspot listing doesn't match")
        return
    
    # 4
    plotSunspotActivity(years, activity)

    # 5
    degree = 10
    [P, T] = prepareInputs(activity, degree)
    print(f"Input list size: {len(P)}")
    print(f"Output list size: {len(T)}")

    # 6
    plotSunspot3D(P, T)

    # 7
    lim = 200
    [Pu, Tu] = [np.array(P[:lim]).reshape(-1, degree), np.array(T[:lim]).reshape(-1, 1)]                       # training data
    [testPu, testTu] = [np.array(P[lim-degree:]).reshape(-1, degree), np.array(T[lim-degree:]).reshape(-1, 1)] # testing data

    # 8
    model = LinearRegression().fit(Pu, Tu)

    # 9
    b = model.intercept_[0]
    print(f"Coefficients: {model.coef_[0]}")
    print(f"Intercept (b): {b}")

    # 10
    Tsu = model.predict(Pu)
    plotPredictionsToReal(years[:lim], Tu, Tsu, 'Training prediction vs training expected Sunspot values')

    testTsu = model.predict(testPu)
    plotPredictionsToReal(years[lim:], testTu, testTsu, 'Test prediction vs test expected Sunspot values')

    # 11
    e = testTu - testTsu
    plotDifference(years[lim:], e)

    # 12
    plotDifferenceHist(e)

    # 13
    mse = MSE(e)
    mad = MAD(e)

    print(f"Mean-Square Error: {mse}")
    print(f"Median Absolute Deviation: {mad}")

    # 14, 15, 16
    learningRate = 0.0000004
    epochs = 1000
    modelAdp = AdaptiveLinearNeuron(learningRate, epochs).fit(Pu, Tu.T[0])

    print(f"Weights after Adaline training: {modelAdp.weight}")
    print(f"MSE for last iteration of Adaline training: {MSE(modelAdp.errors[-1])}")
    print(f"MAD for last iteration of Adaline training: {MAD(modelAdp.errors[-1])}")

    plt.show()
    return

main()