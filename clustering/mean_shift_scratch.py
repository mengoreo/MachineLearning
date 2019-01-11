import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

colors = 10*['g', 'r', 'c', 'b', 'k']


class Mean_Shift:
    def __init__(self, radius=4.3):
        self.radius = radius

    def fit(self, data):
        centroids = {}
        for i in range(len(data)):
            centroids[i] = data[i]

        while True:
            new_centroids = []
            for i in range(len(centroids)):
                in_bandwidth = []
                centroid = centroids[i]
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)
                new_centroid = np.average(in_bandwidth, axis=0)
                # able to be converted to set
                new_centroids.append(tuple(new_centroid))
            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            # update centroids
            centroids = {}
            for i in range(len(uniques)):
                # re-cast tuple to array
                centroids[i] = np.array(uniques[i])

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                    break

            if optimized:
                break

        self.centroids = centroids

    def predict(self, data):
        pass


clf = Mean_Shift()
clf.fit(X)
centroids = clf.centroids
plt.scatter(X[:, 0], X[:, 1], s=150)
for i in centroids:
    plt.scatter(centroids[i][0], centroids[i][1],
                color='k', s=150, marker='*')
plt.show()
