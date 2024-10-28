import numpy as np
import pandas as pd

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