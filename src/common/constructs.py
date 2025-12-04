from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import pickle


class ModelWrapper:
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def train(self, train_X, train_y, test_X=None, test_y=None):
        """
        Fit the model on training data. test_X/test_y are provided in case
        you want early stopping or hyperparameter selection, but you can
        ignore them for simple models.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Return a 1D risk score for each observation. Higher should mean higher
        risk / shorter survival.
        """
        raise NotImplementedError

    def predict_proba(self, X):
        """
        Optional: return a monotone transform of risk scores in (0, 1).
        For survival models this is *not* a calibrated probability.
        """
        raise NotImplementedError


class Preprocessor:
    """
    Interface for preprocessing modules

    Designed to be independent of the model; models see only the output of transform().
    """

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


class IdentityPreprocessor(Preprocessor):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X



@dataclass
class Experiment:
    name: str
    model: ModelWrapper
    T_eval: float
    A_strata: int = 50 # controls precision of causal evaluation
    preprocessor: Preprocessor = field(default_factory=IdentityPreprocessor)
    metadata: Dict[str, Any] | None = None


class ExperimentArtifacts:
    """
    Container for outputs of running an experiment:
      - trained model and preprocessor (frozen copies)
      - arbitrary data arrays / tables
      - scalar metrics
      - a copy of the Experiment config
    """

    def __init__(self, experiment: Optional[Experiment] = None):
        self.experiment: Optional[Experiment] = deepcopy(experiment)
        self.data: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.model: Optional[ModelWrapper] = None
        self.preprocessor: Optional[Preprocessor] = None

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentArtifacts":
        path = Path(path)
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, ExperimentArtifacts):
            raise TypeError(
                f"Loaded object is {type(obj)}, expected ExperimentArtifacts"
            )
        return obj

    def set_model(self, model: ModelWrapper):
        self.model = deepcopy(model)

    def set_preprocessor(self, preprocessor: Preprocessor):
        self.preprocessor = deepcopy(preprocessor)

    def append_data(self, item: dict):
        self.data[item["name"]] = item["data"]

    def append_metric(self, item: dict):
        self.metrics[item["name"]] = item["data"]

    def print_summary(self):
        exp_name = self.experiment.name if self.experiment is not None else "<unnamed>"
        print("=" * 80)
        print(f"Experiment Artifacts Summary: {exp_name}")
        print("=" * 80)
        for name in sorted(self.metrics):
            print(f" - {name}: {self.metrics[name]}")
        print()

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
