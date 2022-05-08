import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizer_v2.adam import Adam
from keras import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold

tf.random.set_seed(1)
np.random.seed(1)

np.set_printoptions(suppress=True)

def main():

    df = pd.read_csv('corrected.csv')
    inputColumns = ['loudness', 'acousticness']
    outputColumn = ['energy']

    X = df[inputColumns].to_numpy()
    Y = df[outputColumn].to_numpy()

    percentToUse = 1
    n = int(len(X) * percentToUse)

    X = X[:n]
    Y = Y[:n]

    K = 10
    kFold = KFold(K)

    resultMSEs = []

    for trainIdx, testIdx in kFold.split(X, Y):
        tf.random.set_seed(1)
        np.random.seed(1)

        model = Sequential()
        model.add(Dense(46, activation='relu'))
        model.add(Dense(1))
        opt = Adam(0.001)
        model.compile(optimizer=opt, loss='mean_squared_error', metrics=['mse'])

        model.fit(X[trainIdx], Y[trainIdx], batch_size=1, epochs=30, verbose=1, validation_split=0.2)

        mse = model.evaluate(X[testIdx], Y[testIdx])[0]

        print(mse)
        resultMSEs.append(mse)

    print(resultMSEs)
    print(np.mean(resultMSEs))

    return

main()