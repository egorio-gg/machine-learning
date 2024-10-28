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

class MyTreeClf:
    def __init__(
            self,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = None,
            criterion: str = 'entropy'
            ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins
        self.criterion = criterion

        self.root = None
        self.leafs_cnt = 0
        self.sum_leafs_val = 0
        self.fi = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyTreeClf class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyTreeClf class: ' + ', '.join(params)

    #best split block
    @staticmethod
    def _gini(vec):
        probs = vec.value_counts() / vec.shape[0]
        return 1 - probs.pow(2).sum()
    
    @staticmethod
    def _entropy(vec):
        probs = vec.value_counts() / vec.shape[0]
        return -1. * (probs * np.log2(probs)).sum()

    def __info_gain(self, p, l_sub, r_sub):
        crt_func = getattr(self, '_' + self.criterion)
        gain = crt_func(p)
        if l_sub.shape[0]:
            gain -= l_sub.shape[0] / p.shape[0] * crt_func(l_sub)
        if r_sub.shape[0]:
            gain -= r_sub.shape[0] / p.shape[0] * crt_func(r_sub)
        return gain 

    def get_best_split(self, X: pd.DataFrame, y: pd.Series):
        best_col, best_treshold, best_gain = None, None, float('-inf')

        for col in X.columns:
            value = X[col]
            tresholds = None
            if self.bins:
                tresholds = self.tresholds[col]
            else:
                col_np = np.sort(np.unique(value))
                tresholds = .5 * (col_np[1:] + col_np[:-1])

            for treshold in tresholds:
                left_y = y[value <= treshold]
                right_y = y[value > treshold]
                gain = self.__info_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_col = col
                    best_treshold = treshold
                    best_gain = gain

        return best_col, best_treshold, best_gain
    #print_tree block
    def print_tree(self, tree: Node = None, indent = '  '):
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)
        else:
            print(f'{tree.col} > {tree.treshold} ? gain = {tree.gain}')
            print('%sleft: ' % (indent), end='')
            self.print_tree(tree.left, indent + indent)
            print('%sright: ' % (indent), end='')
            self.print_tree(tree.right, indent + indent)

    #fit block
    def __conditions(self, depth, n_samples):
        return (depth < self.max_depth) and \
            (n_samples >= self.min_samples_split) and \
            (self.leafs_cnt < self.max_leafs)

    def __build_tree(self, X: pd.DataFrame, y: pd.Series, cur_depth = 0):
        if self.__conditions(cur_depth, X.shape[0]):
            col, treshold, gain = self.get_best_split(X, y)

            if gain > 0:
                self.leafs_cnt += 2 if cur_depth == 0 else 1
                self.fi[col] += X.shape[0] / self.train_size[0] * gain

                idx_l, idx_r = (X[col] <= treshold), (X[col] > treshold)
                X_l, y_l = X[idx_l], y[idx_l]
                X_r, y_r = X[idx_r], y[idx_r]
                sub_l = self.__build_tree(X_l, y_l, cur_depth + 1)
                sub_r = self.__build_tree(X_r, y_r, cur_depth + 1)
                return Node(col, treshold, sub_l, sub_r, gain)
        
        leaf_value = float(y.mean())
        self.sum_leafs_val += leaf_value
        return Node(value=leaf_value)

    def __tresholds_preprocessing(self, col: pd.Series):
        col_np = np.sort(np.unique(col))
        tresholds = .5 * (col_np[1:] + col_np[:-1])

        if not(tresholds.shape[0] <= self.bins - 1):
            _, tresholds = np.histogram(col, self.bins)
            tresholds = tresholds[1:-1]

        return tresholds

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train_size = X.shape
        self.fi = dict(zip(X.columns, [0] * X.shape[0]))

        if self.bins:
            self.tresholds = X.apply(self.__tresholds_preprocessing, axis=0)
        self.root = self.__build_tree(X, y)

    #predict block
    def __predict_one(self, row: pd.Series, tree: Node = None):
        if not tree:
            tree = self.root

        if tree.value is not None:
            return tree.value
        
        if row[tree.col] <= tree.treshold:
            return self.__predict_one(row, tree.left)
        else:
            return self.__predict_one(row, tree.right)

    def predict_proba(self, X: pd.DataFrame):
        return X.apply(self.__predict_one, axis=1)

    def predict(self, X: pd.DataFrame):
        return self.predict_proba(X).apply(lambda x: 1 if x > 0.5 else 0)