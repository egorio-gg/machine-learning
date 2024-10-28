import numpy as np
import pandas as pd

class MyDBSCAN:

    def __init__(self,
                 eps: float = 3,
                 min_samples: int = 3,
                 metric: str = 'euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

        self.data = None
        self.clusters = []
        self.noise = []

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyDBSCAN class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyDBSCAN class: ' + ', '.join(params)

    @staticmethod
    def _euclidean_distance(x: np.ndarray, y: np.ndarray):
        return np.sqrt(np.sum((x - y) ** 2))

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

    def _get_neighbors(self, point: np.ndarray):
        neighbors = []
        for id, candidate in enumerate(self.data):
            if getattr(self, '_' + self.metric + '_distance')(point, candidate) < self.eps:
                neighbors.append(id)
        return neighbors

    def _expand_cluster(self, visited, id, neighbors):
        self.clusters.append([id])
        i = 0

        while i < len(neighbors):
            next_id = neighbors[i]
            if not visited[next_id]:
                visited[next_id] = True
                next_neighbors = self._get_neighbors(self.data[next_id])
                if len(next_neighbors) >= self.min_samples:
                    neighbors += next_neighbors
            cluster_indices = [i for cluster in self.clusters for i in cluster]
            if next_id not in cluster_indices:
                self.clusters[-1].append(next_id)
            i += 1

    def _get_labels(self):
        labels = np.array([0] * self.data.shape[0], dtype=np.int64)

        for i, cluster in enumerate(self.clusters):
            labels[cluster] = i
        
        labels[self.noise] = len(self.clusters)
        return labels
                
    def fit_predict(self, X: pd.DataFrame):
        self.data = X.to_numpy()
        visited_flag = [False] * X.shape[0]
        for id, point in enumerate(self.data):
            if not visited_flag[id]:
                visited_flag[id] = True
                neighbors = self._get_neighbors(point)
                if len(neighbors) < self.min_samples:
                    self.noise.append(id)
                else:
                    self._expand_cluster(visited_flag, id, neighbors)

        self.labels_ = self._get_labels()
        return self.labels_