import numpy as np
import pandas as pd
import random

class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate: float = 0.1,
                 metric: str = None,
                 reg: str = None,
                 l1_coef: float = 0.,
                 l2_coef: float = 0.,
                 sgd_sample = None,
                 random_state: int = 42):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        self.weights = None
        self.best_score = None
        

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyLogReg class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyLogReg class: ' + ', '.join(params)
    
    @staticmethod
    def accuracy(y_true, y_pred):
        return (y_true == y_pred).mean()
    
    @staticmethod
    def precision(y_true, y_pred):
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        return tp / (tp + fp)

    @staticmethod
    def recall(y_true, y_pred):
        tp = (y_true * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        return tp / (tp + fn)

    @staticmethod
    def f1(y_true, y_pred):
        precision_score = MyLogReg().precision(y_true, y_pred)
        recall_score = MyLogReg().recall(y_true, y_pred)
        return 2 * precision_score * recall_score / (precision_score + recall_score)

    @staticmethod
    def roc_auc(y_true, y_score):
        df_roc = pd.DataFrame({'y_true': y_true, 'y_score': y_score})
        df_roc = df_roc.sort_values(by='y_score', ascending=False).reset_index(drop=True)
        df_roc.y_score = df_roc.y_score.apply(lambda x: round(x, 10))

        P = df_roc.y_true.sum()
        N = (1 - df_roc.y_true).sum()
        auc = 0
        for idx, instance in df_roc.iterrows():
            actual = instance['y_true']
            score = instance['y_score']
            if actual:
                continue
            else:
                auc += df_roc.y_true[df_roc.y_score > score].sum()
                auc += 1 / 2 * df_roc.y_true[df_roc.y_score == score].sum()

        return auc / (P * N)

    #regularization
    def l1(self):
        grad = self.l1_coef * np.sign(self.weights)
        return self.l1_coef * np.abs(self.weights).sum(), grad

    def l2(self):
        grad = self.l2_coef * 2 * self.weights
        return self.l2_coef * np.power(self.weights, 2).sum(), grad

    def elasticnet(self):
        l1_loss, l1_grad = self.l1()
        l2_loss, l2_grad = self.l2()
        return l1_loss + l2_loss, l1_grad + l2_grad

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)
        eps = 1e-15
        X_train = X.copy(deep=True).reset_index(drop=True)
        X_train.insert(loc=0, column='_', value=1)
        y_train = y.copy(deep=True).reset_index(drop=True)

        N, M = X_train.shape
        self.weights = np.ones(M)

        for i in range(self.n_iter):
            idx_batch = list(range(N))
            if (self.sgd_sample):
                if isinstance(self.sgd_sample, int):
                    idx_batch = random.sample(range(N), self.sgd_sample)
                if isinstance(self.sgd_sample, float):
                    idx_batch = random.sample(range(N), round(N * self.sgd_sample))
        
            z = X_train @ self.weights
            y_hat = 1 / (1 + np.exp(-z))
            log_loss = -1 / N * (y_train * np.log(y_hat + eps) + \
             (1 - y_train) * np.log(1 - y_hat + eps)).sum()

            grad = 1 / len(idx_batch) * (y_hat[idx_batch] - y_train[idx_batch]) @ X_train.iloc[idx_batch]
            if self.reg:
                reg_loss, reg_grad = getattr(self, self.reg)()
                log_loss += reg_loss
                grad += reg_grad

            if isinstance(self.learning_rate, (int, float)):
                self.weights -= self.learning_rate * grad
            else:
                self.weights -= self.learning_rate(i + 1) * grad

        if self.metric:
            z = X_train @ self.weights
            y_score = 1 / (1 + np.exp(-z))
            if self.metric == 'roc_auc':
                self.best_score = self.roc_auc(y_train, y_score)
            else:
                y_pred = y_score.apply(lambda x: 1 if x > 0.5 else 0)
                self.best_score = getattr(self, self.metric)(y_train, y_pred)

    def get_best_score(self):
        return self.best_score

    def get_coef(self):
        return self.weights[1:]

    def predict_proba(self, X: pd.DataFrame):
        X_test = X.copy(deep=True)
        X_test.insert(loc=0, column='_', value=1)
        z = X_test @ self.weights
        return 1 / (1 + np.exp(-z))

    def predict(self, X: pd.DataFrame):
        y_proba = self.predict_proba(X)
        return y_proba.apply(lambda x: 1 if x > 0.5 else 0)