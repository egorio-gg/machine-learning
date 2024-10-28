import numpy as np
import pandas as pd
import random
import copy

class MyLineReg:

    def __init__(self,
                 n_iter: int = 100,
                 learning_rate = 0.1,
                 metric: str = None,
                 reg: str = None,
                 l1_coef: float = 0,
                 l2_coef: float = 0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.weights = None
        self.best_score = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return '{name}'.format(name=self.__class__.__name__) + ' class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return '{name}'.format(name=self.__class__.__name__) + ' class: ' + ', '.join(params)

    def get_coef(self):
        return self.weights[1:]

    @staticmethod
    def _mse(y_true, y_pred):
        return (y_true - y_pred).pow(2).mean()

    @staticmethod
    def _mae(y_true, y_pred):
        return (y_true - y_pred).abs().mean()

    @staticmethod
    def _rmse(y_true, y_pred):
        return np.sqrt((y_true - y_pred).pow(2).mean())

    @staticmethod
    def _mape(y_true, y_pred):
        return ((y_true - y_pred) / y_true).abs().mean() * 100

    @staticmethod
    def _r2(y_true, y_pred):
        return 1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()

    def l1_reg(self):
        loss = self.l1_coef * self.weights.abs().sum()
        grad = self.l1_coef * self.weights.apply(lambda x: 1 if x >= 0 else -1)
        return grad, loss

    def l2_reg(self):
        loss = self.l2_coef * self.weights.pow(2).sum()
        grad = 2 * self.l2_coef * self.weights
        return grad, loss

    def elasticnet_reg(self):
        l1_grad, l1_loss = self.l1_reg()
        l2_grad, l2_loss = self.l2_reg()
        return l1_grad + l2_grad, l1_loss + l2_loss

    def mse_loss(self, y_true: pd.Series, y_pred: pd.Series):
        return (y_true - y_pred).pow(2).mean()

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        log = lambda iter, loss, metric: f'{iter} | loss: {loss} | {self.metric}: {metric}' \
            if self.metric else f'{iter} | loss: {loss}'

        X_train = X.copy(deep=True).reset_index(drop=True)
        y_train = y.copy(deep=True).reset_index(drop=True)
        X_train.insert(loc=0, column = '__unit__', value=1)
        N, M = X_train.shape

        self.weights = pd.Series(np.ones(M), index=X_train.columns)

        for i in range(self.n_iter):
            y_hat = X_train @ self.weights
            grad = 2 / N * (y_hat - y_train) @ X_train
            loss = self.mse_loss(y_train, y_hat)

            if self.reg:
                grad_reg, loss_reg = getattr(self, self.reg + '_reg')()
                grad += grad_reg
                loss += loss_reg

            if isinstance(self.learning_rate, (int, float)):
                self.weights -= self.learning_rate * grad
            else:
                self.weights -= self.learning_rate(i + 1) * grad

            if verbose and (i == 0 or i % verbose == 0):

                if self.metric:
                    metric = getattr(self, '_' + self.metric)(y_train, y_hat)
                    print(log(i if i > 0 else 'start', loss, metric))
                else:
                    print(log(i if i > 0 else 'start', loss, None))

        if (self.metric):
            self.best_score = getattr(self, '_' + self.metric)(y_train, X_train @ self.weights)

    def get_best_score(self):
        return self.best_score

    def predict(self, X: pd.Series):
        X_test = X.copy(deep=True)
        X_test.insert(loc=0, column = '__unit__', value=1)
        return X_test @ self.weights

class MyKNNReg:
    
    def __init__(self, 
                 k: int = 3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKNNReg class: ' + ' '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKNNReg class: ' + ' '.join(params)

    def euclidean(self, row: pd.Series):
        return (self.X_train - row).pow(2).sum(axis=1).pow(.5)

    def chebyshev(self, row: pd.Series):
        return (self.X_train - row).abs().max(axis=1)

    def manhattan(self, row: pd.Series):
        return (self.X_train - row).abs().sum(axis=1)

    def cosine(self, row: pd.Series):
        norms_train = self.X_train.pow(2).sum(axis=1).pow(.5)
        norm_row = np.sqrt(row.pow(2).sum())
        return 1 - (self.X_train @ row) / (norms_train * norm_row)

    def _uniform_vote(self, dist: pd.Series):
        return self.y_train[dist.sort_values().head(self.k).index].mean()

    def _rank_vote(self, dist: pd.Series, proba=False):
        target_k = self.y_train[dist.sort_values().head(self.k).index].reset_index(drop=True)
        weights = (1 / (target_k.index + 1)).values
        return (target_k * weights / weights.sum()).sum()

    def _distance_vote(self, dist: pd.Series, proba=False):
        target_k = self.y_train[dist.sort_values().head(self.k).index].reset_index(drop=True)
        dist_k = dist.sort_values().head(self.k).reset_index(drop=True)
        weights = 1 / dist_k
        return (target_k * weights / weights.sum()).sum()

    def __predict_unit(self, row: pd.Series):
        dist = getattr(self, self.metric)(row)
        return getattr(self, '_' + self.weight + '_vote')(dist)
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X.copy(deep=True)
        self.y_train = y.copy(deep=True)
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame):
        return X.apply(self.__predict_unit, axis=1)

class Node:
    def __init__(
            self,
            col: str = None,
            treshold: float = None,
            left = None,
            right = None,
            gain = None,
            value: float = None
            ):
        #decision nodes
        self.col = col
        self.treshold = treshold
        self.left = left
        self.right = right
        self.gain = gain

        #leaves nodes
        self.value = value

class MyTreeReg:
    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs = 20,
            bins: int = None
            ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.leafs_cnt = 0
        self.bins = bins

        self.fi = dict()
        self.root = None
        self.sum_leafs_val = 0

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyTreeReg class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyTreeReg class: ' + ', '.join(params)

    def __mse(self, vec: pd.Series):
        return 1/vec.shape[0] * (vec - vec.mean()).pow(2).sum()

    def __mse_gain(self, p: pd.Series, left_sub: pd.Series, right_sub: pd.Series):
        gain = 0
        if p.shape[0]:
            gain = self.__mse(p)
        else:
            return None

        if left_sub.shape[0]:
            gain -= left_sub.shape[0] / p.shape[0] * self.__mse(left_sub)

        if right_sub.shape[0]:
            gain -= right_sub.shape[0] / p.shape[0] * self.__mse(right_sub)

        return gain

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_col_name, best_treshold, best_gain = None, None, float('-inf')

        for col in X.columns:
            values = X[col]
            col_np = np.sort(np.unique(values))
            tresholds = None
            if self.bins:
                tresholds = self.tresholds[col]
            else:
                tresholds = .5 * (col_np[1:] + col_np[:-1])

            for treshold in tresholds:
                left_y = y[values <= treshold]
                right_y = y[values > treshold]
                gain = self.__mse_gain(y, left_y, right_y)

                if gain and gain > best_gain:
                    best_col_name = col
                    best_treshold = treshold
                    best_gain = gain

        return best_col_name, best_treshold, best_gain

    def print_tree(self, tree: Node = None, indent = '  '):
        if tree is None:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(f'{tree.col} > {tree.treshold} ? gain = {tree.gain}')
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


    def __conditions(self, depth, num_samples):
        return (depth < self.max_depth) and \
            (num_samples >= self.min_samples_split) and \
            (self.leafs_cnt < self.max_leafs)


    def __build_tree(self, X: pd.DataFrame, y: pd.Series, cur_depth = 0):

        if self.__conditions(cur_depth, X.shape[0]):
            self.leafs_cnt += 2 if cur_depth == 0 else 1
            col, treshold, gain = self.get_best_split(X, y)
            self.fi[col] += X.shape[0] / self.train_size[0] * gain

            left_idx = (X[col] <= treshold)
            right_idx = (X[col] > treshold)
            X_left, y_left = X[left_idx], y[left_idx]
            X_right, y_right = X[right_idx], y[right_idx]

            left_sub = self.__build_tree(X_left, y_left, cur_depth + 1)
            right_sub = self.__build_tree(X_right, y_right, cur_depth + 1)
            return Node(col, treshold, left_sub, right_sub, gain)

        leaf_val = float(y.mean())
        self.sum_leafs_val += leaf_val
        return Node(value=leaf_val)

    def __tresholds_preprocessing(self, col: pd.Series):
        col_np = np.sort(np.unique(col))
        tresholds = .5 * (col_np[1:] + col_np[:-1])
        if not(tresholds.shape[0] <= self.bins - 1):
            _, tresholds = np.histogram(col, self.bins)
            tresholds = tresholds[1:-1]
        return tresholds

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train_size = X.shape
        self.fi = dict(zip(X.columns, [0]*X.shape[1]))

        if self.bins:
            self.tresholds = X.apply(self.__tresholds_preprocessing, axis=0)
        self.root = self.__build_tree(X, y)

    def __predict_one(self, row: pd.Series, tree: Node = None):
        if not tree:
            tree = self.root

        if tree.value is not None:
            return tree.value

        if row[tree.col] <= tree.treshold:
            return self.__predict_one(row, tree.left)
        else:
            return self.__predict_one(row, tree.right)

    def predict(self, X: pd.DataFrame):
        return X.apply(self.__predict_one, axis=1)

class MyBaggingReg:

    def __init__(self,
                 estimator = None,
                 n_estimators = 10,
                 max_samples = 1.0,
                 random_state = 42,
                 oob_score = None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        self.oob_score_ = None
        self.estimators = []

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBaggingReg class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBaggingReg class: ' + ', '.join(params)

    @staticmethod
    def mae(y_true, y_pred):
        return (y_true - y_pred).abs().mean()

    @staticmethod
    def mse(y_true, y_pred):
        return (y_true - y_pred).pow(2).mean()

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt((y_true - y_pred).pow(2).mean())

    @staticmethod
    def mape(y_true, y_pred):
        return 100 * ((y_true - y_pred) / y_true).abs().mean()

    @staticmethod
    def r2(y_true, y_pred):
        return 1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)
        
        indices = X.index.to_list()
        N, M = X.shape
        bootstrap_idx = []
        oob_idx = []
        oob_idx_ = set()
        y_pred = pd.Series([0] * len(indices), index=indices)
        count = pd.Series([0] * len(indices), index=indices)

        for _ in range(self.n_estimators):
            idx = random.choices(indices, k=round(N * self.max_samples))
            bootstrap_idx += [idx]
            oob_idx += [list(set(indices) - set(idx))]
            oob_idx_ = oob_idx_ | (set(indices) - set(idx))

        for idx in bootstrap_idx:
            model = copy.deepcopy(self.estimator)
            model.fit(X.loc[idx, :], y[idx])
            self.estimators += [model]

        for i, idx in enumerate(oob_idx):
            count[idx] += 1
            y_pred[idx] += self.estimators[i].predict(X.loc[idx, :])
        
        oob_idx_ = list(oob_idx_)
        y_pred = y_pred[oob_idx_] / count[oob_idx_]
        self.oob_score_ = getattr(self, self.oob_score)(y.loc[oob_idx_], y_pred)


    def predict(self, X: pd.DataFrame):
        y_pred = None

        for model in self.estimators:
            if y_pred is None:
                y_pred = model.predict(X)
                continue

            y_pred += model.predict(X)

        return y_pred / self.n_estimators