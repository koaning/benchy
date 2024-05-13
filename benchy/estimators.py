from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, substring):
        self.substring = substring

    def fit(self, X, y=None):
        self.cols_ = [c for c in X.columns if self.substring not in c.lower()]
        return self
    
    def transform(self, X, y=None): 
        return X.select(*self.cols)