import numpy as np
import pandas as pd
import random
import copy

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
        self.fi_N = None
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
            col, treshold, gain = self.get_best_split(X, y)

            if gain > 0:
                self.leafs_cnt += 2 if cur_depth == 0 else 1
                N = self.fi_N if self.fi_N else self.train_size[0]
                self.fi[col] += X.shape[0] / N * gain

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

    #for gradient boosting
    def _update_leaves(self,
                       X: pd.DataFrame,
                       y: pd.Series,
                       tree: Node = None,
                       loss: str = 'MSE',
                       reg = 0.):
        if not tree:
            tree = self.root

        if tree.value is not None:
            if loss == 'MAE':
                tree.value = y.median() + reg
            else:
                tree.value = y.mean() + reg
            return

        values = X[tree.col]
        l_idx = (values <= tree.treshold)
        r_idx = (values > tree.treshold)
        self._update_leaves(X.loc[l_idx, :], y[l_idx], tree.left, loss, reg)
        self._update_leaves(X.loc[r_idx, :], y[r_idx], tree.right, loss, reg)

class MyBoostReg:

    def __init__(self,
                 n_estimators: int = 10,
                 learning_rate: float = 0.1,
                 max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_leafs: int = 20,
                 bins: int = 16,
                 loss: str = 'MSE',
                 metric: str = None,
                 max_features: float = 0.5,
                 max_samples: float = 0.5,
                 random_state: int = 42,
                 reg: float = 0.1):

        #GradientBoosting parametrs
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        #MSE, MAE, RMSE, R2, MAPE
        self.metric = metric
        self.reg = reg

        #Trees parametrs
        self.max_depth = max_depth
        self.min_samples_split = 2
        self.max_leafs = 20
        self.bins = bins

        #Fit parametrs
        self.pred_0 = None
        self.trees = []
        self.best_score = None
        self.fi = dict()

        #SGD
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBoostReg class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBoostReg class: ' + ', '.join(params)

    @staticmethod
    def mse_grad_loss(y_true, y_pred):
        mse = (y_pred - y_true).pow(2).sum()
        grad_mse = 2 * (y_pred - y_true)
        return grad_mse, mse

    @staticmethod
    def mae_grad_loss(y_true, y_pred):
        mae = (y_pred - y_true).abs().sum()
        grad_mae = (y_pred - y_true) / (y_pred - y_true).abs()
        return grad_mae, mae

    #metrics
    @staticmethod
    def mse(y_true, y_pred) -> float:
        return (y_true - y_pred).pow(2).mean()

    @staticmethod
    def rmse(y_true, y_pred) -> float:
        return np.sqrt((y_true - y_pred).pow(2).mean())

    @staticmethod
    def mae(y_true, y_pred) -> float:
        return (y_true - y_pred).abs().mean()

    @staticmethod
    def mape(y_true, y_pred) -> float:
        return 100 * ((y_true - y_pred) / y_true).abs().mean()

    @staticmethod
    def r2(y_true, y_pred) -> float:
        return 1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()

    def calc_score(self, X: pd.DataFrame, y: pd.Series):
        y_pred = self.predict(X)

        if self.metric is None:
            return getattr(self, self.loss.lower())(y, y_pred)
        else:
            return getattr(self, self.metric.lower())(y, y_pred)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_eval: pd.DataFrame = None,
            y_eval: pd.Series = None,
            early_stopping: int = None,
            verbose: int = None):

        random.seed(self.random_state)
        N, M = X.shape
        leaves_count = 0
        self.fi = dict(zip(X.columns.to_list(), [0.] * M))

        if (self.loss == 'MSE'):
            self.pred_0 = y.mean()
        elif (self.loss == 'MAE'):
            self.pred_0 = y.median()

        y_pred = pd.Series([self.pred_0] * y.shape[0], index=y.index.to_list())

        #early stopping preparation
        stop_count = 0
        prev_metric_eval = float('inf') if self.metric != 'R2' else float('-inf')
        for i in range(self.n_estimators):
            cols_idx = random.sample(X.columns.to_list(),
                                     round(M * self.max_features))
            rows_idx = random.sample(X.index.to_list(),
                                     round(N * self.max_samples))

            grad, loss = getattr(self, (self.loss).lower() + '_grad_loss')(y.loc[rows_idx], y_pred.loc[rows_idx])
            tree = MyTreeReg(self.max_depth,
                             self.min_samples_split,
                             self.max_leafs,
                             self.bins)
            tree.fi_N = N
            tree.fit(X.loc[rows_idx, cols_idx], -1. * grad)
            tree._update_leaves(X.loc[rows_idx, cols_idx],
                                (y - y_pred).loc[rows_idx],
                                loss = self.loss,
                                reg = self.reg * leaves_count)
            self.trees += [tree]

            leaves_count += tree.leafs_cnt
            if verbose and i % verbose == 0:
                if self.metric:
                    metric = getattr(self, self.metric.lower())(y, y_pred)
                    print(f'{i}. Loss[{self.loss}]: {loss} | {self.metric}: {metric}', end='')
                else:
                    print(f'{i}. Loss[{self.loss}]: {loss}', end='')

                if early_stopping:
                    print(f' | Eval_metric: {metric_eval}', end='')
                print()

            if isinstance(self.learning_rate, (int, float)):
                y_pred = y_pred + self.learning_rate * tree.predict(X)
            else:
                y_pred = y_pred + self.learning_rate(i + 1) * tree.predict(X)

            if early_stopping:
                metric_eval = self.calc_score(X_eval, y_eval)
                if (metric_eval >= prev_metric_eval and self.metric != 'R2') or \
                    (metric_eval <= prev_metric_eval and self.metric == 'R2'):
                    early_stopping_count += 1
                else:
                    early_stopping_count = 0
                    prev_metric_eval = metric_eval

                if early_stopping_count == early_stopping:
                    self.trees = self.trees[:-early_stopping]
                    break

        if early_stopping:
            self.best_score = self.calc_score(X_eval, y_eval)
        else:
            self.best_score = self.calc_score(X, y)

        for tree in self.trees:
            for col, val in tree.fi.items():
                self.fi[col] += val

    def predict(self, X: pd.DataFrame):
        y_pred = pd.Series([self.pred_0] * X.shape[0], index=X.index.to_list())

        for i, tree in enumerate(self.trees):
            if isinstance(self.learning_rate, (int, float)):
                y_pred = y_pred + self.learning_rate * tree.predict(X)
            else:
                y_pred = y_pred + self.learning_rate(i + 1) * tree.predict(X)

        return y_pred