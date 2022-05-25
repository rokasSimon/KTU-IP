from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
from sklearn import datasets

def displayTargetGroups():

    iris = datasets.load_iris()
    x = iris.data[:, :4]
    y = iris.target

    plt.figure()

    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolor='k')
    plt.xlabel("Taurėlapio ilgis")
    plt.ylabel("Taurėlapio plotis")

    plt.figure()

    plt.scatter(x[:, 2], x[:, 3], c=y, edgecolor='k')
    plt.xlabel("Vainiklapio ilgis")
    plt.ylabel("Vainiklapio plotis")

    return

def main():

    iris = datasets.load_iris()
    data = iris.data[:, :4]

    #displayTargetGroups()

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
            kms = KMeans(k, init='random', algorithm='elkan', max_iter=5, random_state=1)
            kms.fit(x)

            labels = kms.predict(x)
            inertia.append(kms.inertia_)

            print('------------------------------------------------------------------------------------------------------')
            print(f'Cluster k = {k}:')
            print(f'Centroids: {kms.cluster_centers_}')
            print(f'Inertia: {kms.inertia_}')
            print('------------------------------------------------------------------------------------------------------')

            silhouetteAvg = silhouette_score(x, labels)
            silhouettes = silhouette_samples(x, labels)

            fig, (ax1, ax2) = plt.subplots(1, 2)

            yLower = 10
            for i in range(k):
                clusterSilhouetteValues = silhouettes[labels == i]
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

            colors = cm.nipy_spectral(labels.astype(float) / k)
            ax2.scatter(x[:, 0], x[:, 1], s=30, c=colors)

            ax2.scatter(
                kms.cluster_centers_[:, 0],
                kms.cluster_centers_[:, 1],
                marker="o",
                c="white",
                s=200,
                edgecolor="k",
            )

            for i, c in enumerate(kms.cluster_centers_):
                ax2.scatter(c[0], c[1], marker="$%d$" % i, s=100, edgecolor="k")

            ax2.set_xlabel(dataSetColumnNames[xi])
            ax2.set_ylabel(dataSetColumnNames[yi])

            plt.suptitle(f'K-vidurkių metodo vizualizacija su k = {k}')

        ax.plot(ks, inertia, label=f'Column pair [{xi+1} {yi+1}]')

    ax.set_ylabel('Inercija')
    ax.set_xlabel('Klasteriai k')
    ax.legend()
    plt.show()

    return

main()