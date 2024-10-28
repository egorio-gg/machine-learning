import numpy as np
import pandas as pd

class Cluster:

    def __init__(self, points: np.ndarray, labels: list):
        self.points = points
        self.labels = labels

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'Cluster: ' + ', '.join(params)

    def __rept__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'Cluster: ' + ', '.join(params)

class MyAgglomerative:

    def __init__(self,
                 n_clusters: int = 3,
                 metric: str = 'euclidean'):
        self.n_clusters = n_clusters
        self.metric = metric

        ''' fit_prefict parametrs '''
        self.clusters = None
        self.data = None
        self.labels_ = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return f'{self.__class__.__name__} class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return f'{self.__class__.__name__} class: ' + ', '.join(params)

    @staticmethod
    def _euclidean_distance(x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.power(x - y, 2).sum())

    @staticmethod
    def _chebyshev_distance(x: np.ndarray, y: np.ndarray):
        return np.max(np.abs(x - y))

    @staticmethod
    def _manhattan_distance(x: np.ndarray, y: np.ndarray):
        return np.abs(x - y).sum()

    @staticmethod
    def _cosine_distance(x: np.ndarray, y: np.ndarray):
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return 1 - (x * y).sum() / (norm_x * norm_y)
    
    def _init_clusters(self):
        return {id: Cluster(point.reshape(1, -1), [id]) for id, point in enumerate(self.data)}

    def _find_cluster_centroid(self, cluster: np.ndarray):
        return cluster.mean(axis=0)
        
    def _find_closest_clusters(self):
        min_dist = float('inf')
        closest_clusters = None
        clusters_id = list(self.clusters.keys())

        for i, cluster_i in enumerate(clusters_id[:-1]):
            centroid_i = self._find_cluster_centroid(self.clusters[cluster_i].points)
            for j, cluster_j in enumerate(clusters_id[i+1:]):
                centroid_j = self._find_cluster_centroid(self.clusters[cluster_j].points)
                dist = getattr(self, '_' + self.metric + '_distance')(centroid_i, centroid_j)

                if dist < min_dist:
                    min_dist = dist
                    closest_clusters = (cluster_i, cluster_j)
        
        return closest_clusters

    def _merge_and_form_clusters(self, ci_id, cj_id):
        new_points = np.concatenate((self.clusters[ci_id].points, 
                                    self.clusters[cj_id].points), axis=0)
        new_labels = self.clusters[ci_id].labels + self.clusters[cj_id].labels
        new_clusters = {0: Cluster(new_points, new_labels)}
        for cluster_id in self.clusters.keys():
            if (cluster_id == ci_id) or (cluster_id == cj_id):
                continue
            new_clusters[len(new_clusters.keys())] = self.clusters[cluster_id]
        return new_clusters
    
    def _calc_labels(self):
        labels = np.zeros(self.data.shape[0], dtype=np.int64)
        for key, cluster in self.clusters.items():
            labels[cluster.labels] = key
        return labels

    def fit_predict(self, X: pd.DataFrame):
        self.data = X.to_numpy()
        self.clusters = self._init_clusters()
        
        while len(self.clusters.keys()) > self.n_clusters:
            closest_clusters = self._find_closest_clusters()
            self.clusters = self._merge_and_form_clusters(*closest_clusters)

        self.labels_ = self._calc_labels()
        return self.labels_