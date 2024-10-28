import numpy as np
import pandas as pd

class MyPCA():

    def __init__(self,
                 n_components: int = 3):
        self.n_components = n_components
        self.data = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyPCA class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyPCA class: ' + ', '.join(params)

    def _norm_data(self):
        return self.data - self.data.mean(axis=0)

    def _calc_cov_matrix(self):
        self.data = self._norm_data()
        _, M = self.data.shape
        cov_matrix = np.zeros(shape=(M, M),
                              dtype=np.float64)
        for i in range(M):
            for j in range(M):
                cov_matrix[i, j] = (self.data[:, i] * self.data[:, j]).mean()
        return cov_matrix

    def fit_transform(self, X: pd.DataFrame):
        self.data = X.to_numpy()
        cov_matrix = self._calc_cov_matrix()
        eig_val, eig_vec = np.linalg.eigh(cov_matrix)
        sorted_idx = np.argsort(eig_val)[::-1]
        sorted_eig_val = eig_val[sorted_idx]
        sorted_eig_vec = eig_vec[:, sorted_idx]
        return pd.DataFrame(self.data @ sorted_eig_vec[:, :self.n_components])