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