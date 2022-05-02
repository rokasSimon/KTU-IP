import pandas as pd
from keras import Sequential
from keras.layers import Dense
import keras.optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def main():

    df = pd.read_csv('normalized.csv')
    inputColumns = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'key', 'mode']
    outputColumn = ['Popularity']

    X = df[inputColumns].to_numpy()
    Y = df[outputColumn].to_numpy()

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.3, random_state=0)

    model = Sequential()
    model.add(Dense(2, activation='sigmoid', input_dim=X.shape[1]))
    model.add(Dense(1, activation='sigmoid'))
    opt = keras.optimizers.sgd_experimental.SGD(0.01)
    #model.compile(optimizer='SGD', loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    #print(model.get_weights())

    res = model.fit(xtrain, ytrain, batch_size=10, epochs=50, verbose=1, validation_split=0.2)

    #print(model.get_weights())

    return

main()