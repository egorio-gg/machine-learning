import numpy as np
import pandas as pd
import random

class MyLineReg():

    def __init__(
            self,
            n_iter: int = 100,
            learning_rate = 0.1,
            metric: str = None, #'mae', 'mse', 'mape', 'rmse', 'r2'
            reg: str = None, #'l1', 'l2', 'elasticnet'
            l1_coef: float = 0.,
            l2_coef: float = 0.,
            sgd_sample = None,
            random_state: int = 42
            ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.score = None
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

    def __str__(self) -> str:
        params = [f'{key} = {value}' for key, value in self.__dict__.items()]
        return 'MyLineReg class: ' + ', '.join(params)

    def get_coef(self):
        return self.weights[1:]

    @staticmethod
    def mae(y_true: np.array, y_pred: np.array):
        return np.abs((y_true - y_pred)).mean()

    @staticmethod
    def mse(y_true: np.array, y_pred: np.array):
        return np.power((y_true - y_pred), 2).mean()

    @staticmethod
    def rmse(y_true: np.array, y_pred: np.array):
        return np.sqrt(np.power((y_true - y_pred), 2).mean())

    @staticmethod
    def mape(y_true: np.array, y_pred: np.array):
        return 100 * np.abs(((y_true - y_pred) / y_true)).mean()

    @staticmethod
    def r2(y_true: np.array, y_pred: np.array):
        return 1 - np.power((y_true - y_pred), 2).sum() / np.power((y_true - y_true.mean()), 2).sum()

    def _l1(self):
        loss = self.l1_coef * np.abs(self.weights).sum()
        grad = self.l1_coef * np.sign(self.weights)
        return loss, grad

    def _l2(self):
        loss = self.l2_coef * np.power(self.weights, 2).sum()
        grad = self.l2_coef * 2 * self.weights
        return loss, grad

    def _elasticnet(self):
        l1 = self._l1()
        l2 = self._l2()
        loss = l1[0] + l2[0]
        grad = l1[1] + l2[1] 
        return loss, grad
    
    def loss_grad_calc(self, X: np.array, y_true: np.array, 
                       y_pred: np.array, idx_batch: np.array):
        loss = np.power((y_true - y_pred), 2).mean()
        X_batch = X[idx_batch, :]
        y_true_batch = y_true[idx_batch]
        y_pred_batch = y_pred[idx_batch]
        grad = 2 / y_true_batch.shape[0] * (y_pred_batch - y_true_batch) @ X_batch
        reg_loss, reg_grad = 0, 0
        if (self.reg):
            reg_loss, reg_grad = getattr(self, '_' + self.reg)()
        return loss + reg_loss, grad + reg_grad

    def get_best_score(self):
        return self.score

    def fit(self, X, y, verbose=False):
        random.seed(self.random_state)
        #N - number of samples, M - number of features
        N, M = X.shape
        X_train = X.to_numpy()
        y_train = y.to_numpy()
        X_train = np.concatenate((np.ones((N, 1)), X), axis=1)
        self.X = X_train
        self.y = y_train

        self.weights = np.ones(M+1)
        for i in range(1, self.n_iter + 1):

            idx_batch = np.arange(0, N)
            if (self.sgd_sample):
                if isinstance(self.sgd_sample, int):
                    idx_batch = random.sample(range(N), self.sgd_sample)
                if isinstance(self.sgd_sample, float):
                    idx_batch = random.sample(range(N),
                                                round(N * self.sgd_sample))

            y_pred = X_train @ self.weights
            loss, grad = self.loss_grad_calc(X_train, y_train, y_pred, idx_batch)

            if isinstance(self.learning_rate, (int, float)):
                self.weights -= self.learning_rate * grad
            else:
                self.weights -= self.learning_rate(i) * grad

            #logs
            if self.metric:
                self.score = getattr(self, self.metric)(y, y_pred)

            if verbose and i % verbose == 0:
                if self.metric:
                    print(f'{i} | loss: {loss} | {self.metric}: {self.score}')
                else:
                    print(f'{i} | loss: {loss}')

    def predict(self, X):
        X_pred = X.to_numpy()
        X_pred = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X_pred @ self.weights