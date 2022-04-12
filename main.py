from sklearn.cluster import KMeans
from scipy.stats import norm
from csv import reader

import matplotlib.pyplot as plt
import numpy as np
import pyransac3d


def open_csv(name, nl="\n", dl=","):
    punkty = []
    with open(name, newline=nl) as csvfile:
        csvreader = reader(csvfile, delimiter=dl)
        for xx, yy, zz in csvreader:
            punkty.append([float(xx), float(yy), float(zz)])
    return punkty


chmura = open_csv("Cylinder.xyz")


clusterer = KMeans(n_clusters=3)


X = np.array(chmura)
y_pred = clusterer.fit_predict(X)

red = y_pred == 0
green = y_pred == 1
blue = y_pred == 2

figure_2 = plt.figure()
ax_2 = figure_2.add_subplot(projection='3d')

ax_2.scatter(X[red, 0], X[red, 1], X[red, 2], c="red")
ax_2.scatter(X[green, 0], X[green, 1], X[green, 2], c="green")
ax_2.scatter(X[blue, 0], X[blue, 1], X[blue, 2], c="blue")
plt.show()

if __name__ == '__main__':
    cloud_points_read = np.array(read_csv("Cylinder.xyz"))

    plane = pyransac3d.Plane()
    best_eq, best_inliers = plane.fit(cloud_points_read, thresh=0.01, minPoints=50, maxIteration=200)

    print(f'best equation Ax+By+Cz+D:{best_eq}')
    print(f'best inliers:{best_inliers}')
