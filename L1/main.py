from math import ceil
import numpy as np
import matplotlib.pyplot as plt
from numpy import std
import pandas as pd
import seaborn as sns

def replaceEmptyValues(inputFileName: str, outputFileName: str):
    df: pd.DataFrame = pd.read_csv(inputFileName)

    popularityMedian = df["Popularity"].median()
    keyMode = int(df["key"].mode()[0])

    df['Popularity'] = df['Popularity'].fillna(popularityMedian)
    df['key'] = df['key'].fillna(keyMode)

    df.to_csv(outputFileName)

    return

def normalizeValues(inputFileName: str, outputFileName: str, columns: list[str]):
    df: pd.DataFrame = pd.read_csv(inputFileName)

    df[columns] = (df[columns] - df[columns].min()) / (df[columns].max() - df[columns].min())

    df.to_csv(outputFileName)

    return

def outputLinearStats(df: pd.DataFrame, columns: list[str]):
    all_columns = df[columns]

    for (colName, data) in all_columns.iteritems():
        n = len(data)
        nMissing = data.isnull().sum()
        nMissingProc = round(nMissing / n * 100, 2)
        nUnique = len(data.unique())
        minVal = data.min()
        maxVal = data.max()
        (firstQuantile, thirdQuantile) = data.quantile([0.25, 0.75])
        mean = data.mean()
        med = data.median()
        stddev = data.std()

        print(f"{colName}:")
        print(f"    Number of rows: {n}")
        print(f"    Missing values: {nMissingProc}%")
        print(f"    Cardinality: {nUnique}")
        print(f"    Minimum value: {minVal}")
        print(f"    Maximum value: {maxVal}")
        print(f"    .25 quantile: {firstQuantile}")
        print(f"    .75 quantile: {thirdQuantile}")
        print(f"    Mean: {mean}")
        print(f"    Median: {med}")
        print(f"    Standard Deviation: {stddev}")

    all_columns.hist(bins = ceil(1 + 3.22 * np.log(len(all_columns))))

    return

def outputCategoricalStats(df: pd.DataFrame, columns: list[str]):
    all_columns = df[columns]

    for (colName, data) in all_columns.iteritems():
        n = len(data)
        nMissing = data.isnull().sum()
        nMissingProc = round(nMissing / n * 100, 2)
        nUnique = len(data.unique())

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
        print(f"    Missing values: {nMissingProc}%")
        print(f"    Cardinality: {nUnique}")
        print(f"    Mode 1: {mod1}")
        print(f"    Frequency of Mode 1: {modFreq1}")
        print(f"    Mode 1, %: {modeProc1}")
        print(f"    Mode 2: {mod2}")
        print(f"    Frequency of Mode 2: {modFreq2}")
        print(f"    Mode 2, %: {modeProc2}")

    all_columns.hist(bins = ceil(1 + 3.22 * np.log(len(all_columns))))

    return

def linearScatterMatrix(df: pd.DataFrame, columns: list[str]):
    all_columns = df[columns]

    pd.plotting.scatter_matrix(all_columns)

    return

def linearScatterPlots(df: pd.DataFrame, pairs: list[list[str]]):
    for pair in pairs:
        firstCol = df[pair[0]]
        secondCol = df[pair[1]]

        data = pd.DataFrame(data = {pair[0]: firstCol, pair[1]: secondCol})

        data.plot.scatter(x = pair[0], y = pair[1], s = 10)

    return

def categoricalBarPlots(df: pd.DataFrame):
    # only a single graph is diplayed at a time, so has to be run multiple times...
    #df.groupby(['key']).size().plot(kind = 'bar')

    #majorKeys = df.loc[df['mode'] == 1]
    #majorKeys.groupby(['key']).size().plot(kind = 'bar')

    #minorKeys = df.loc[df["mode"] == 0]
    #minorKeys.groupby(['key']).size().plot(kind = 'bar')

    return

def outputCovariance(df: pd.DataFrame, columns: list[str]):

    data = df[columns]

    print(data.cov())

    return

def outputCorrelation(df: pd.DataFrame, columns: list[str]):

    data = df[columns]

    corr = data.corr()
    print(corr)

    sns.heatmap(corr, cmap='Blues', annot=True)

    return

def outputLinearAndCategoricalDataBoxPlots(df: pd.DataFrame):

    df['Popularity'].plot.box()
    df.boxplot(by='key', column=['Popularity'], grid=True)
    df.boxplot(by='mode', column=['Popularity'], grid=True)

    #df['energy'].plot.box()
    #df.boxplot(by='key', column=['energy'], grid=True)
    #df.boxplot(by='mode', column=['energy'], grid=True)

    return

def main():
    startingDataFile = 'test.csv'
    correctedDataFile = 'corrected.csv'
    normalizedDataFile = 'normalized.csv'

    #df: pd.DataFrame = pd.read_csv(startingDataFile)

    #replaceEmptyValues(startingDataFile, correctedDataFile)
    df: pd.DataFrame = pd.read_csv(correctedDataFile)
    #df: pd.DataFrame = pd.read_csv(normalizedDataFile)

    correctedLinearColumnNames = ['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo']
    correctedCategoricalColumnNames = ['key', 'mode']

    # Starting input
    #outputLinearStats(df, ['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_in min/ms'])
    #outputCategoricalStats(df, ['Artist Name', 'Track Name', 'key', 'mode', 'time_signature'])

    # For corrected data
    #outputLinearStats(df, correctedLinearColumnNames)
    #outputCategoricalStats(df, correctedCategoricalColumnNames)

    #linearScatterPlots(df, [
    #    ['loudness', 'energy'],
    #    ['valence', 'danceability'],
    #    ['energy', 'tempo']
    #])
    #linearScatterMatrix(df, correctedLinearColumnNames)
    #categoricalBarPlots(df)

    #outputCovariance(df, correctedLinearColumnNames)
    #outputCorrelation(df, correctedLinearColumnNames)

    #normalizeValues(correctedDataFile, normalizedDataFile, ['Popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'key', 'mode'])

    outputLinearAndCategoricalDataBoxPlots(df)

    plt.show()

    return

main()