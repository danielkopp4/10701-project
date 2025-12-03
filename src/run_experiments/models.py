import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest as SKRandomSurvivalForest


from ..common import ModelWrapper


class Cox(ModelWrapper):
    def __init__(self, alpha: float = 0.0, **kwargs):
        super().__init__(name="CoxPH")
        self.alpha = alpha
        self.kwargs = kwargs

        self.estimator = CoxPHSurvivalAnalysis(alpha=self.alpha, **self.kwargs)
        self._fitted = False

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("CoxPHWrapper: model has not been trained yet.")

    def train(self, train_X, train_y, test_X=None, test_y=None):
        self.estimator.fit(train_X, train_y)
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        scores = self.estimator.predict(X)
        return np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        scores = self.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict_survival_function(self, X):
        self._check_fitted()
        return self.estimator.predict_survival_function(X)

class GradientBoosting(ModelWrapper):
    def __init__(self, learning_rate: float = 0.1, n_estimators: int = 100, **kwargs):
        super().__init__(name="GradientBoostingCoxPH")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs

        self.estimator = GradientBoostingSurvivalAnalysis(
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            **self.kwargs
        )
        self._fitted = False

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("GradientBoostingCoxWrapper: model has not been trained yet.")

    def train(self, train_X, train_y, test_X=None, test_y=None):
        self.estimator.fit(train_X, train_y)
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        scores = self.estimator.predict(X)
        return np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        scores = self.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))
    
    def predict_survival_function(self, X):
        self._check_fitted()
        return self.estimator.predict_survival_function(X)
    
class RandomSurvivalForest(ModelWrapper):
    def __init__(
        self,
        n_estimators: int = 500,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        max_features: str | int | float | None = "sqrt",
        n_jobs: int = -1,
        random_state: int = 0,
    ):
        super().__init__(name="RandomSurvivalForest")
        self.estimator = SKRandomSurvivalForest(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        self._fitted = False

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("RandomSurvivalForestWrapper: model has not been trained yet.")

    def train(self, train_X, train_y, test_X=None, test_y=None):
        self.estimator.fit(train_X, train_y)
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        expected_time = self.estimator.predict(X)
        return -np.asarray(expected_time, dtype=float)

    def predict_proba(self, X):
        scores = self.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict_survival_function(self, X):
        self._check_fitted()
        return self.estimator.predict_survival_function(X)
