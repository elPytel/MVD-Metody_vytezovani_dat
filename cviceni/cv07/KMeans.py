import numpy as np
import random

def euclidean_dist(v1, v2):
    return np.sqrt(np.sum((v2-v1)**2, axis=0))

def manhattan_dist(v1, v2):
    return np.sum(abs(v2-v1), axis=0)

class KMeans:
    def __init__(self, n_clusters, metric=manhattan_dist):
        self.metric = metric
        self.k = n_clusters

    def generate_centroids(self):
        return np.array(random.choices(self.data, k = self.k))

    def index_of_nearest_centroid_from(self, point):
        distances = np.zeros(self.cluster_centers_.shape[0])
        for index, centroid in enumerate(self.cluster_centers_):
            distances[index] = self.metric(point, centroid)
        return np.argmin(distances)

    def criterion(self, last):
        #print("Criterion: ", np.sum(abs(self.metric(self.cluster_centers_, last))))
        return np.sum(abs(self.metric(self.cluster_centers_, last)))

    def fit(self, data):
        self.data = data
        self.cluster_centers_ = self.generate_centroids()
        self.labels_ = np.zeros(self.data.shape[0])

        last = self.cluster_centers_**2
        while self.criterion(last) > 0.1:
            # cluster data
            for index, point in enumerate(self.data):
                self.labels_[index] = self.index_of_nearest_centroid_from(point)

            # update centroids
            last = self.cluster_centers_.copy()
            for index, centroid in enumerate(self.cluster_centers_):
                indexes = np.argwhere(self.labels_==index)
                points = self.data[indexes,:].squeeze()
                self.cluster_centers_[index] = np.mean(points, axis=0)
        return self.labels_