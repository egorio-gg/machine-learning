import numpy as np
import pandas as pd
import random

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

class MyForestReg:
    def __init__(
            self,
            n_estimators: int = 10,
            max_features: float = 0.5,
            max_samples: float = 0.5,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = 16,
            random_state: int = 42,
            oob_score: str = None
                 ):
        #forest params
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_samples = max_samples

        #trees params
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        #random params
        self.random_state = random_state

        #another
        self.trees = []
        self.leafs_cnt = 0
        self.fi = dict()
        self.oob_score = oob_score
        self.oob_score_ = None
        self.bootstrap_idx = []

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyForestReg class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyForestReg class: ' + ', '.join(params)

    #metrics
    @staticmethod
    def mse(y_true: pd.Series, y_pred: pd.Series):
        return 1 / y_true.shape[0] * (y_true - y_pred).pow(2).sum()

    @staticmethod
    def mae(y_true, y_pred):
        return 1 / y_true.shape[0] * (y_true - y_pred).abs().sum()
    
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(MyForestReg.mse(y_true, y_pred))

    @staticmethod
    def mape(y_true, y_pred):
        return 100 / y_true.shape[0] * ((y_true - y_pred) / y_true).abs().sum()

    @staticmethod
    def r2(y_true, y_pred):
        return 1 - (y_true - y_pred).pow(2).sum() / (y_true - y_true.mean()).pow(2).sum()

    #fit
    def __oob_score_calc(self, X: pd.DataFrame, y: pd.Series):
        oob_prob = np.zeros_like(y, dtype=np.float64)
        oob_count = np.zeros_like(y, dtype=np.int64)

        for i, tree in enumerate(self.trees):
            oob_idx = np.setxor1d(range(X.shape[0]), self.bootstrap_idx[i])
            oob_prob[oob_idx] += tree.predict(X.iloc[oob_idx, :]).to_numpy()
            oob_count[oob_idx] += 1

        validate = oob_count > 0
        oob_prob = oob_prob[validate]
        oob_count = oob_count[validate]
        y_pred = oob_prob / oob_count
        y_true = y[validate]
        
        return getattr(self, self.oob_score)(y_true, y_pred)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)
        self.fi = dict(zip(X.columns, [0]*X.shape[1]))
        for i in range(self.n_estimators):
            N, M = X.shape
            cols_idx = random.sample(X.columns.to_list(), round(self.max_features * M))
            rows_idx = random.sample(range(X.shape[0]), round(self.max_samples * N))
            self.bootstrap_idx += [rows_idx]

            tree = MyTreeReg(max_depth=self.max_depth,
                             min_samples_split=self.min_samples_split,
                             max_leafs=self.max_leafs,
                             bins=self.bins)
            tree.fi_N = N
            tree.fit(X.loc[rows_idx, cols_idx], y[rows_idx])
            self.leafs_cnt += tree.leafs_cnt
            self.trees += [tree]

        for tree in self.trees:
            for key, value in tree.fi.items():
                self.fi[key] += value

        self.oob_score_ = self.__oob_score_calc(X, y)

    #predict
    def predict(self, X: pd.DataFrame):
        predictions = []
        for i, tree in enumerate(self.trees):
            y_pred = tree.predict(X)
            y_pred.name = f'tree_{i}'
            predictions += [y_pred]

        df_predict = pd.concat(predictions, axis=1)
        return df_predict.mean(axis=1)