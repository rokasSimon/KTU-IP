import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn_som.som import SOM
from sklearn import datasets

def read_data():
    result = pd.read_csv('Wholesale_customers_data.csv');
    return result

def main():
    #Take data from iris
    iris = datasets.load_iris()
    data = iris.data[:, :4]

    dataSetIndices = [
        [0, 1],
        [2, 3],
    ]

    dataSetColumnNames = [
        'Taurėlapio ilgis',
        'Taurėlapio plotis',
        'Vainiklapio ilgis',
        'Vainiklapio plotis'
    ]

    ks = np.arange(2, 8, 1)

    mfig, ax = plt.subplots()
    
    for indices in dataSetIndices:
        [xi, yi] = indices

        x = data[:, xi]
        y = data[:, yi]

        x = np.column_stack((x, y))

        inertia = []

        for k in ks:
            som = SOM(m=1, n=k, dim= 2)
            som.fit(x)
            inertia.append(som.inertia_)

            print('------------------------------------------------------------------------------------------------------')
            print(f'Cluster k = {k}:')
            print(f'Centroids: {som.cluster_centers_}')
            print(f'Inertia: {som.inertia_}')
            print('------------------------------------------------------------------------------------------------------')

            predictions = som.predict(x)

            silhouetteAvg = silhouette_score(x, predictions)
            silhouettes = silhouette_samples(x, predictions)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            yLower = 10
            for i in range(k):
                clusterSilhouetteValues = silhouettes[predictions == i]
                clusterSilhouetteValues.sort()

                clusterSize = clusterSilhouetteValues.shape[0]
                yUpper = yLower + clusterSize

                color = cm.nipy_spectral(float(i) / k)

                ax1.fill_betweenx(
                    np.arange(yLower, yUpper),
                    0,
                    clusterSilhouetteValues,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                ax1.text(-0.05, yLower + 0.5 * clusterSize, str(i))
                yLower = yUpper + 10

            ax1.set_xlabel("Silueto koeficientų reikšmės")
            ax1.set_ylabel("Klasteris")

            ax1.axvline(x=silhouetteAvg, color="red", linestyle="--")

            colors = cm.nipy_spectral(predictions.astype(float) / k)
            ax2.scatter(x[:, 0], x[:, 1], s=30, c=colors)

            ax2.scatter(
                som.cluster_centers_[0][:, 0],
                som.cluster_centers_[0][:, 1],
                marker="o",
                c="white",
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(som.cluster_centers_[0]):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, s=100, edgecolor="k")

            ax2.set_xlabel(dataSetColumnNames[xi])
            ax2.set_ylabel(dataSetColumnNames[yi])

            plt.suptitle(f'SOM metodo vizualizacija su k = {k}')

        ax.plot(ks, inertia, label=f'Column pair [{xi+1} {yi+1}]')

    ax.set_ylabel('Inercija')
    ax.set_xlabel('Klasteriai k')
    ax.legend()
    plt.show()

    return
main()