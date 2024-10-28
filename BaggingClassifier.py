import numpy as np
import pandas as pd
import random
import copy

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

class MyKNNClf:

    def __init__(self,
                 k: int = 3,
                 metric: str = 'euclidean',
                 weight: str = 'uniform'):
        self.k = k
        self.metric = metric
        self.weight = weight

        self.train_size = None

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKNNClf class: ' + ' '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyKNNClf class: ' + ' '.join(params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.X_train = X.copy(deep=True)
        self.y_train = y.copy(deep=True)
        self.train_size = X.shape

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

    def _uniform_vote(self, dist: pd.Series, proba=False):
        prob = self.y_train[dist.sort_values().head(self.k).index].mean()
        if proba:
            return prob
        else:
            return 1 if prob >= .5 else 0

    def _rank_vote(self, dist: pd.Series, proba=False):
        target_k = self.y_train[dist.sort_values().head(self.k).index].reset_index(drop=True)
        weights = (1 / (target_k.index + 1)).values.sum()
        weight_0 = (1 / (target_k[target_k == 0].index + 1)).values.sum()
        weight_1 = (1 / (target_k[target_k == 1].index + 1)).values.sum()
        q_0 = weight_0 / weights
        q_1 = weight_1 / weights

        if proba:
            return q_1
        else:
            return 1 if q_1 >= q_0 else 0

    def _distance_vote(self, dist: pd.Series, proba=False):
        target_k = self.y_train[dist.sort_values().head(self.k).index].reset_index(drop=True)
        dist_k = dist.sort_values().head(self.k).reset_index(drop=True)
        weights = (1 / dist_k).sum()
        weight_0 = (1 / dist_k[target_k == 0]).sum()
        weight_1 = (1 / dist_k[target_k == 1]).sum()
        q_0 = weight_0 / weights
        q_1 = weight_1 / weights

        if proba:
            return q_1
        else:
            return 1 if q_1 >= q_0 else 0

    def __predict_unit(self, row: pd.Series, proba=False):
        dist = getattr(self, self.metric)(row)

        return getattr(self, '_' + self.weight + '_vote')(dist, proba)

    def predict_proba(self, X: pd.DataFrame):
        return X.apply(self.__predict_unit, args=(True,), axis=1)

    def predict(self, X: pd.DataFrame):
        return X.apply(self.__predict_unit, args=(False,), axis=1)

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

class MyBaggingClf:
    def __init__(self,
                 estimator = None,
                 n_estimators = 10,
                 max_samples = 1.0,
                 random_state = 42,
                 oob_score: str = None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.oob_score = oob_score

        self.oob_score_ = None
        self.estimators = []

    def __str__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBaggingClf class: ' + ', '.join(params)

    def __repr__(self):
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return 'MyBaggingClf class: ' + ', '.join(params)

    #metrics
    @staticmethod
    def accuracy_score(y_true: pd.Series, y_pred: pd.Series):
        return (y_true == y_pred).mean()

    @staticmethod
    def precision_score(y_true: pd.Series, y_pred: pd.Series):
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        return tp / (tp + fp)

    @staticmethod
    def recall_score(y_true: pd.Series, y_pred: pd.Series):
        tp = (y_true * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        return tp / (tp + fn)

    @staticmethod
    def f1_score(y_true: pd.Series, y_pred: pd.Series):
        tp = (y_true * y_pred).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def roc_auc_score(y_true: pd.Series, y_score: pd.Series):
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

    def fit(self, X: pd.DataFrame, y: pd.Series):
        random.seed(self.random_state)

        indices = X.index.to_list()
        bootstrap_idx = []
        oob_idx = []
        oob_idx_intersection = set()

        N, M = X.shape
        for _ in range(self.n_estimators):
            idx = random.choices(indices, k=round(N * self.max_samples))
            bootstrap_idx += [idx]
            oob = list(set(indices) - set(idx))
            oob_idx += [oob]
            oob_idx_intersection = oob_idx_intersection | set(oob)

        oob_idx_intersection = list(oob_idx_intersection)

        for i, idx in enumerate(bootstrap_idx):
            model = copy.deepcopy(self.estimator)
            model.fit(X.loc[idx, :], y[idx])
            self.estimators += [model]

        y_score = pd.Series([0] * len(oob_idx_intersection), index=oob_idx_intersection)
        count = y_score.copy(deep=True)

        for i, idx in enumerate(oob_idx):
            model = self.estimators[i]
            y_score[idx] += model.predict_proba(X.loc[idx, :])
            count[idx] += 1

        y_score = y_score / count

        if (self.oob_score):
            if (self.oob_score == 'roc_auc'):
                self.oob_score_ = getattr(self, self.oob_score + '_score')(y[oob_idx_intersection], y_score)
            else:
                y_pred = y_score.apply(lambda x: 1 if x > 0.5 else 0)
                self.oob_score_ = getattr(self, self.oob_score + '_score')(y[oob_idx_intersection], y_pred)

    def predict(self, X: pd.DataFrame, type='mean'):
        y_pred = None

        for model in self.estimators:
            predict = model.predict_proba if type == 'mean' else model.predict
            if y_pred is None:
                y_pred = predict(X)
                continue

            y_pred += predict(X)

        if type == 'mean':
            y_pred /= self.n_estimators
            y_pred = y_pred.apply(lambda x: 1 if x > 0.5 else 0)

        else:
            y_pred /= self.n_estimators
            y_pred = y_pred.apply(lambda x: 1 if x >= 0.5 else 0)

        return y_pred

    def predict_proba(self, X: pd.DataFrame):
        y_pred = None

        for model in self.estimators:
            if y_pred is None:
                y_pred = model.predict_proba(X)
                continue

            y_pred += model.predict_proba(X)

        return y_pred / self.n_estimators