import numpy as np
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest as SKRandomSurvivalForest
from sksurv.functions import StepFunction
from sksurv.linear_model.coxph import BreslowEstimator
from sksurv.svm import FastSurvivalSVM
from sksurv.base import SurvivalAnalysisMixin

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
    def __init__(
            self,
            learning_rate: float = 0.05,
            n_estimators: int = 100,
            max_depth: int = 3,
            min_samples_split: int = 50,
            min_samples_leaf: int = 50,
            subsample: float = 0.6,
            max_features: str | int | float | None = "sqrt",
            random_state: int = 0,
            **kwargs,
             ):
        super().__init__(name="GradientBoosting")
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.kwargs = kwargs

        self.estimator = GradientBoostingSurvivalAnalysis(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            **kwargs,
        )
        self._fitted = False

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("GradientBoosting: model has not been trained yet.")

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
        n_estimators: int = 400,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_depth: int | None = None,
        # max_depth: int | None = 8,
        # max_features: str | int | float | None = "sqrt",
        max_features: str | int | float | None = 0.8,
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
            raise RuntimeError("RandomSurvivalForest: model has not been trained yet.")

    def train(self, train_X, train_y, test_X=None, test_y=None):
        self.estimator.fit(train_X, train_y)
        self._fitted = True
        return self

    def predict(self, X):
        self._check_fitted()
        expected_time = self.estimator.predict(X)
        return np.asarray(expected_time, dtype=float)

    def predict_proba(self, X):
        scores = self.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict_survival_function(self, X):
        self._check_fitted()
        return self.estimator.predict_survival_function(X)
        
class SVM(ModelWrapper, SurvivalAnalysisMixin):
    def __init__(self, alpha: float = 1.0, rank_ratio: float = 1.0, **kwargs):
        super().__init__(name="SVM")
        self.alpha = alpha
        self.rank_ratio = rank_ratio
        self.kwargs = kwargs

        self.estimator = FastSurvivalSVM(
            alpha=self.alpha,
            rank_ratio=self.rank_ratio,
            **self.kwargs
        )
        self._fitted = False
        self.breslow = None

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("SVM: model has not been trained yet.")

    def train(self, train_X, train_y, test_X=None, test_y=None):
        self.estimator.fit(train_X, train_y)
        risk_scores = self.estimator.predict(train_X)

        event_name = train_y.dtype.names[0]
        time_name = train_y.dtype.names[1]

        events = train_y[event_name]
        times = train_y[time_name]

        self.breslow = BreslowEstimator().fit(risk_scores, events, times)
        
        self._fitted = True
        return self

    def predict(self, X):
        """Predict risk scores"""        
        self._check_fitted()
        scores = self.estimator.predict(X)
        return np.asarray(scores, dtype=float)

    def predict_proba(self, X):
        scores = self.predict(X)
        return 1.0 / (1.0 + np.exp(-scores))
    
    def predict_survival_function(self, X, return_array=False):
        self._check_fitted()
        risk_scores = self.predict(X)
        surv_funcs = self.breslow.get_survival_function(risk_scores)

        if return_array:
            unique_times = self.breslow.unique_times_
            surv_array = np.row_stack([fn(unique_times) for fn in surv_funcs])
            return surv_array
            
        return surv_funcs
