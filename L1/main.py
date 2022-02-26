from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from numpy import std
import pandas as pd

def normalizeDuration():
    df = pd.read_csv('test.csv')

    for i, row in df.iterrows():
        newValue = float(row['duration_in min/ms'])
        if newValue < 100.0:
            newValue *= 60000.0

        df.at[i, 'duration_in min/ms'] = newValue

    df.to_csv('test.csv')

    return

# Columns: Popularity, Danceability, Energy, Loudness, Speechiness, Instrumentalness, Liveness, Tempo, Time Signature
def get_linear_data_stats(df: pd.DataFrame, columns: list[str]):

    all_columns = df[columns]

    for (colName, data) in all_columns.iteritems():
        n = len(data)
        n_missing = data.isnull().sum()
        n_unique = len(data.unique())
        min_val = data.min()
        max_val = data.max()
        (first_quantile, third_quantile) = data.quantile([0.25, 0.75])
        av = data.sum() / n
        med = data.median()
        stddev = data.std()

        print(f"{colName}:")
        print(f"    Number of rows: {n}")
        print(f"    Number of missing values: {n_missing}")
        print(f"    Cardinality: {n_unique}")
        print(f"    Minimum value: {min_val}")
        print(f"    Maximum value: {max_val}")
        print(f"    .25 quantile: {first_quantile}")
        print(f"    .75 quantile: {third_quantile}")
        print(f"    Average: {av}")
        print(f"    Median: {med}")
        print(f"    Standard Deviation: {stddev}")

    all_columns.hist(bins = ceil(1 + 3.22 * np.log(n)))

    return

def get_categorical_data_stats(df: pd.DataFrame, columns: list[str]):

    all_columns = df[columns]

    for (colName, data) in all_columns.iteritems():
        n = len(data)
        n_missing = data.isnull().sum()
        n_unique = len(data.unique())

        vc = data.value_counts()
        modFreq1 = vc.iloc[0]
        modFreq2 = vc.iloc[1]

        vcidx = vc.index.tolist()
        mod1 = vcidx[0]
        mod2 = vcidx[1]

        modeProc1 = round(modFreq1 / n * 100, 2)
        modeProc2 = round(modFreq2 / n * 100, 2)
        
        print(f"{colName}:")
        print(f"    Number of rows: {n}")
        print(f"    Number of missing values: {n_missing}")
        print(f"    Cardinality: {n_unique}")
        print(f"    Mode 1: {mod1}")
        print(f"    Frequency of Mode 1: {modFreq1}")
        print(f"    Mode 1, %: {modeProc1}")
        print(f"    Mode 2: {mod2}")
        print(f"    Frequency of Mode 2: {modFreq2}")
        print(f"    Mode 2, %: {modeProc2}")

    all_columns.hist(bins = ceil(1 + 3.22 * np.log(n)))

    return

def main():

    df = pd.read_csv('test.csv')

    get_linear_data_stats(df, ['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'instrumentalness', 'liveness', 'tempo'])
    get_categorical_data_stats(df, ['time_signature'])

    # testing
    plt.show()

    return

main()