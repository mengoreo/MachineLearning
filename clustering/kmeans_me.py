import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

# plt.scatter(X[:, 0], X[:, 1], s=150)
# plt.show()


colors = ['g', 'r', 'c', 'b', 'k'] * 10


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=333):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        # randomly chose k features as centroids
        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):

            # initiate clusters
            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = []

            # find the closest features to the clusters
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                # classify to the closest one
                cluster = distances.index(min(distances))
                self.clusters[cluster].append(featureset)

            # Store the last centorids
            # Using dict to avoid prev_centroids changeing along with self.centroids
            prev_centroids = dict(self.centroids)

            # redefine the new clusters
            for cluster in self.clusters:
                # pass
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)

            optimized = True
            for c in self.centroids:
                prev_centroid = prev_centroids[c]
                curr_centroid = self.centroids[c]
                if np.sum((curr_centroid - prev_centroid)/prev_centroid*100.0) > self.tol:
                    # print
                    optimized = False
            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        cluster = distances.index(min(distances))
        return cluster

clf = K_Means()
clf.fit(X)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker='o', color='k', s=150, linewidths=5)

for cluster in clf.clusters:
    color = colors[cluster]
    for featureset in clf.clusters[cluster]:
        plt.scatter(featureset[0], featureset[1], marker='x', c=color, s=150, linewidths=5)

unknowns = np.array([[1, 2],
                     [3, 4],
                     [5, 5],
                     [8, 9]])

for unknown in unknowns:
    cluster = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker='d', c=colors[cluster], s=150, linewidths=5)
plt.show()
