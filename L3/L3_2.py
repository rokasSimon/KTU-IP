import pandas as pd
from keras import Sequential

def main():

    df = pd.read_csv('normalized.csv')

    inputCols = df.iloc[1, 5:]

    print(inputCols)

    return