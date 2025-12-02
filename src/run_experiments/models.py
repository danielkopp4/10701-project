import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis

from .constructs import ModelWrapper


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
