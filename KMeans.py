import numpy as np
import pandas as pd
import random
import copy

class MyKMeans:

    def __init__(
            self,
            n_clusters: int = 3,
            max_iter: int = 10,
            n_init: int = 3,
            random_state: int = 42
            ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

        self.cluster_centers_ = None
        self.inertia_ = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKMeans class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKMeans class: ' + ', '.join(params)

    def clusters_distribution(self, X: np.ndarray, centers: np.ndarray):
        N, _ = X.shape
        M, _ = centers.shape

        dist_matrix = np.zeros(shape=(N, M), dtype=np.float64)
        for idx, center in enumerate(centers):
            dist_matrix[:, idx] = np.sqrt(np.power(X - center, 2).sum(axis=1))

        return dist_matrix.argmin(axis=1)

    def update_centers(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray):
        new_centers = centers.copy()
        for i in range(self.n_clusters):
            X_i = X[y == i]
            if X_i.size != 0:
                new_centers[i] = X_i.mean(axis=0)
        return new_centers

    def wcss_score(self, X: np.ndarray, y: np.ndarray, centers: np.ndarray):
        wcss = 0.
        for i, center in enumerate(centers):
            X_i = X[y == i]
            if X_i.size != 0:
                wcss += np.power(X_i - center, 2).sum()
        return wcss
    
    def fit(self, X: pd.DataFrame):
        np.random.seed(seed=self.random_state)

        X_train = X.to_numpy(dtype=np.float64)
        N, M = X_train.shape
        min_ = X_train.min(axis=0)
        max_ = X_train.max(axis=0)

        centers_cluster = []
        for i in range(self.n_init):
            centers = np.random.uniform(min_, max_, size=(self.n_clusters, M))
            prev_centers = centers.copy()
            for j in range(self.max_iter):
                target = self.clusters_distribution(X_train, centers)
                centers = self.update_centers(X, target, centers)
                if np.isclose(centers, prev_centers).any():
                    break
                else:
                    prev_centers = centers.copy()
            centers_cluster += [centers]
        
        best_wcss = float('inf')
        best_centers = None

        for _, centers_ in enumerate(centers_cluster):
            target = self.clusters_distribution(X_train, centers_)
            wcss = self.wcss_score(X_train, target, centers_)
            if wcss < best_wcss:
                best_wcss = wcss
                best_centers = centers
        
        self.cluster_centers_ = best_centers
        self.inertia_ = best_wcss

    def predict(self, X: pd.DataFrame):
        X_test = X.to_numpy()
        target = self.clusters_distribution(X, self.cluster_centers_)
        return pd.Series(target, index=X.index.to_list())