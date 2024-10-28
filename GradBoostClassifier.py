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
        self.fi = dict(zip(X.columns, [0.0]*X.shape[1]))

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

    def _update_leaves(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            probs: pd.Series,
            tree: Node = None,
            reg_: float = 0.
            ):

        if not tree:
            tree = self.root

        if tree.value is not None:
            tree.value = (y - probs).sum() / (probs * (1 - probs)).sum() + reg_
            return

        values = X[tree.col]
        l_idx = (values <= tree.treshold)
        r_idx = (values > tree.treshold)
        self._update_leaves(X.loc[l_idx, :], y[l_idx], probs[l_idx], tree.left, reg_)
        self._update_leaves(X.loc[r_idx, :], y[r_idx], probs[r_idx], tree.right, reg_)

class MyBoostClf:
    ''' Class for Gradient Boosting Classifier '''

    def __init__(
            self,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            max_depth: int = 5,
            min_samples_split: int = 2,
            max_leafs: int = 20,
            bins: int = 16,
            metric: str = None,
            max_features: float = 0.5,
            max_samples: float = 0.5, 
            random_state: int = 42,
            reg: float = 0.1
            ):
        ''' Gradient Boosting Classifier parametrs '''
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.metric = metric

        ''' Decision Tree Classifier parametrs '''
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_leafs = max_leafs
        self.bins = bins

        '''Stohastic Gradient Boosting parametrs '''
        self.max_features = max_features
        self.max_samples = max_samples
        self.random_state = random_state

        ''' Regularization '''
        self.reg = reg

        ''' Fit parametrs '''
        self.pred_0 = None
        self.trees = []
        self.best_score = None
        self.fi = dict()

    def __str__(self):
        ''' function for print info about the instance by calling a print() '''
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBoostClf class: ' + ', '.join(params)

    def __repr__(self):
        ''' function for print info about the instance by calling a instance '''
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBoostClf class: ' + ', '.join(params)

    def __to_log_odds(self, probs: pd.Series):
        return np.log(probs / (1 - probs) + 1e-15)

    def __to_probs(self, log_odds: pd.Series):
        return np.exp(log_odds) / (1 + np.exp(log_odds))

    ''' Metrics '''

    @staticmethod
    def accuracy_score(y_true, y_pred):
        return (y_true == y_pred).mean()

    @staticmethod
    def precision_score(y_true, y_pred):
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        return tp / (tp + fp)

    @staticmethod
    def recall_score(y_true, y_pred):
        tp = (y_true * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        return tp / (tp + fn)

    @staticmethod
    def f1_score(y_true, y_pred):
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * recall * precision / (recall + precision)

    @staticmethod
    def roc_auc_score(y_true, y_score):
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
    
    @staticmethod
    def loss_calc(y_true, log_odds):
        return -(y_true * log_odds - np.log(1 + np.exp(log_odds)) + 1e-15).mean()

    def calc_score(self, X: pd.DataFrame, y: pd.Series):
        y_score = self.predict_proba(X)
        loss = self.loss_calc(y, self.__to_log_odds(y_score))

        if self.metric is None:
            return loss

        if self.metric != 'roc_auc':
            y_pred = y_score.apply(lambda x: 1 if x > 0.5 else 0)
            return getattr(self, self.metric + '_score')(y, y_pred)
        else:
            return self.roc_auc_score(y, y_score)

    def print_logs(self, iter: int, y: pd.Series, log_odds: pd.Series):
        loss = self.loss_calc(y, log_odds)
        if (self.metric):
            if self.metric != 'roc_auc':
                y_pred = self.__to_probs(log_odds).apply(lambda x: 1 if x > 0.5 else 0)
                metric_val = getattr(self, self.metric + '_score')(y, y_pred)
            else:
                y_score = self.__to_probs(log_odds)
                metric_val = self.roc_auc_score(y, y_score)
            return f'{iter}. Loss: {loss} | {self.metric.capitalize()}: {metric_val}'
        else:
            return f'{iter}. Loss: {loss}'

    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            X_eval: pd.DataFrame = None,
            y_eval: pd.Series = None,
            early_stopping: int = None, 
            verbose: int = None):
        random.seed(self.random_state)
        
        N, M = X.shape
        self.fi = dict(zip(X.columns.to_list(), [0.] * M))

        p_0 = y.mean()
        self.pred_0 = np.log(p_0 / (1 - p_0) + 1e-15)
        log_odds = pd.Series([self.pred_0] * N, index=y.index.to_list())
        leaves_count = 0

        prev_metric_eval = float('-inf') if self.metric else float('inf')
        early_stopping_count = 0
        for i in range(self.n_estimators):
            cols_idx = random.sample(X.columns.to_list(), k=round(M * self.max_features))
            rows_idx = random.sample(X.index.to_list(), k=round(N * self.max_samples))

            probs = self.__to_probs(log_odds)
            if verbose and i % verbose == 0:
                print(self.print_logs(i, y, log_odds))
            anti_grad = y - probs
            tree = MyTreeReg(self.max_depth,
                             self.min_samples_split,
                             self.max_leafs,
                             self.bins)
            tree.fi_N = N
            tree.fit(X.loc[rows_idx, cols_idx], anti_grad[rows_idx])
            leaves_count += tree.leafs_cnt
            tree._update_leaves(X.loc[rows_idx, cols_idx], 
                                y[rows_idx], 
                                probs[rows_idx],
                                reg_ = self.reg * leaves_count)
            self.trees += [tree]
            if isinstance(self.learning_rate, (int, float)):
                log_odds = log_odds + self.learning_rate * tree.predict(X)
            else:
                log_odds = log_odds + self.learning_rate(i + 1) * tree.predict(X)

            if early_stopping:
                metric_eval = self.calc_score(X_eval, y_eval)
                if (metric_eval <= prev_metric_eval and self.metric) or \
                    (metric_eval >= prev_metric_eval and (self.metric is None)):
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

    def predict_proba(self, X: pd.DataFrame):
        y_pred = pd.Series([self.pred_0] * X.shape[0], index=X.index.to_list())
        for idx, tree in enumerate(self.trees):
            if isinstance(self.learning_rate, (int, float)):
                y_pred = y_pred + self.learning_rate * tree.predict(X)
            else:
                y_pred = y_pred + self.learning_rate(idx + 1) * tree.predict(X)

        return self.__to_probs(y_pred)

    def predict(self, X: pd.DataFrame):
        return self.predict_proba(X).apply(lambda x: 1 if x > 0.5 else 0)