from typing import List, Set

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from .constructs import Preprocessor


# IDs / outcome columns we never want as features
ALWAYS_EXCLUDE_COLS: Set[str] = {
    "SEQN",
    "time",
    "event",
    "mortstat",
    "died",
    "died_cvd",
    "died_cancer",
    "permth_int",
    "permth_exm",
    "eligstat",
    # “cheating” markers: direct proxies for hazard (synthetic only)
    "disease",
    "prob_disease",
}

# columns potentially causally downstream of weight / adiposity:
DOWNSTREAM_OF_WEIGHT_COLS: Set[str] = {
    # Blood pressure / pulse
    "BPXSY", "BPXDI", "BPXPULS",
    "BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4",
    "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4",

    # Lipids
    "LBXTC", "LBDHDD", "LBDLDL", "LBXTR",

    # Glucose metabolism
    "LBXGLU", "LBXGH", "LBXIN",

    # CBC
    "LBXWBCSI", "LBXLYPCT", "LBXMOPCT", "LBXNEPCT", "LBXEOPCT",
    "LBXBAPCT", "LBXRBCSI", "LBXHGB", "LBXHCT", "LBXMCVSI", "LBXMCHSI",
    "LBXRDW", "LBXPLTSI", "LBXNRBC",

    # Comprehensive metabolic panel
    "LBXSATSI", "LBXSASSI", "LBXSAPSI", "LBXSGTSI", "LBXSAL", "LBXSTP",
    "LBXSTB", "LBXSBU", "LBXSCR", "LBXSUA", "LBXSGL", "LBXSCA", "LBXSPH",
    "LBXSKSI", "LBXSNASI", "LBXSCLSI", "LBXSC3SI", "LBXSIR",

    # Urinary markers
    "URXUCR", "URXUMA",

    # Diagnoses / downstream clinical variables
    "DIQ010", "DIQ160", "DIQ170", "DIQ172", "DIQ050", "DIQ070", "DIQ230",
    "MCQ080", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F",
    "MCQ160L", "MCQ160M", "MCQ160N",
    "CDQ001", "CDQ002", "CDQ003", "CDQ004", "CDQ005", "CDQ006", "CDQ008", "CDQ010",

    # Dataset-level outcome flags
    "diabetes", "hyperten"
}

class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, clip_min: float = -5.0, clip_max: float = 5.0):
        self.clip_min = clip_min
        self.clip_max = clip_max

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.clip_min, self.clip_max)



class NHANESPreprocessor(Preprocessor):
    """
    - filters out ID/outcome columns
    - optionally removes downstream-of-weight variables
    - imputes missing values
    - scales features
    """

    # TODO: add option to encode categorical variables differently
    # TODO: add option to create missingness indicators
    # TODO: add option to do more advanced imputation (e.g. KNN, MICE)
    # TODO: add option to only include "easy to capture" features (intake survey only)
    # TODO: add option to include interaction terms, polynomial features, 
    #       transformations of weight / height to get a better BMI representation

    def __init__(
        self,
        exclude_downstream: bool = True,
        exclude_all: bool = False,
        impute_strategy: str = "median",
    ):
        self.exclude_downstream = exclude_downstream
        self.exclude_all = exclude_all
        self.impute_strategy = impute_strategy

        self.feature_cols_: List[str] | None = None
        self.pipeline_: Pipeline | None = None

    def _select_feature_columns(self, X) -> List[str]:
        cols = list(X.columns)
        exclude = set(ALWAYS_EXCLUDE_COLS)
        if self.exclude_all:
            # exclude everything except for BMI
            exclude |= {c for c in cols if c != "BMXBMI"}
        elif self.exclude_downstream:
            exclude |= DOWNSTREAM_OF_WEIGHT_COLS
        feature_cols = [c for c in cols if c not in exclude]
        if not feature_cols:
            raise ValueError(
                "NHANESPreprocessor: no features left after exclusion. "
                "Check ALWAYS_EXCLUDE_COLS / DOWNSTREAM_OF_WEIGHT_COLS."
            )
        return feature_cols

    def fit(self, X, y=None):
        self.feature_cols_ = self._select_feature_columns(X)
        X_selected = X[self.feature_cols_]

        self.pipeline_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=self.impute_strategy)),
                ("scaler", StandardScaler()),
                ("clipper", ClipTransformer(clip_min=-5.0, clip_max=5.0)),
            ]
        )
        self.pipeline_.fit(X_selected)
        return self

    def transform(self, X):
        if self.feature_cols_ is None or self.pipeline_ is None:
            raise RuntimeError("call fit before transform")
        X_selected = X[self.feature_cols_]
        X_proc = self.pipeline_.transform(X_selected)
        return X_proc
