import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sksurv.util import Surv
import os
from pathlib import Path
from typing import Optional, Tuple

def load_nhanes_survival(
    csv_path: Path | str,
    time_col: str = "time",
    event_col: str = "event",
    id_col: str = "SEQN",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify_events: bool = True,
    extra_exclude_cols: Optional[list[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Load the cleaned NHANES dataset for survival analysis.

    Returns (train_X, train_y, test_X, test_y) where:
    X are pandas DataFrames of covariates
    y are sksurv.util.Surv structured arrays with fields ('event', 'time')
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"NHANES survival dataset not found at: {csv_path}")

    # load data
    df = pd.read_csv(csv_path)
    n_raw = len(df)

    # Basic presence checks
    for col in (time_col, event_col):
        if col not in df.columns:
            raise KeyError(f"Required column `{col}` not found in {csv_path}")

    # final survival filtering
    mask_valid = df[time_col].notna() & df[event_col].notna()
    mask_valid &= df[time_col] > 0

    df = df[mask_valid].copy()
    n_valid = len(df)

    if n_valid == 0:
        raise ValueError("No rows with valid (time > 0, non-missing event).")

    # Ensure event is 0/1
    unique_events = set(df[event_col].dropna().unique())
    if not unique_events.issubset({0, 1}):
        raise ValueError(f"`{event_col}` must be in {{0,1}}, got: {unique_events}")

    # build covariate matrix X, excluding leakage columns
    base_exclude = {
        time_col,
        event_col,
        "mortstat",    # raw mortality status
        "died",        # derived death flag
        "permth_exm",  # follow-up time (exam-based)
        "permth_int",  # follow-up time (interview-based)
        "eligstat",    # linkage eligibility
    }

    # Do not use ID as a feature by default
    if id_col is not None:
        base_exclude.add(id_col)

    if extra_exclude_cols:
        base_exclude.update(extra_exclude_cols)

    exclude_cols = [c for c in base_exclude if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns remaining after exclusion.")

    X_all = df[feature_cols].copy()

    # build surv object for survival labels
    time_all = df[time_col].astype(float).to_numpy()
    event_all = df[event_col].astype(int).astype(bool).to_numpy()

    y_all = Surv.from_arrays(event=event_all, time=time_all)

    # train/test split
    stratify = event_all if stratify_events else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv('config.env')
    csv_path = Path(os.getenv('PROCESSED_DATA_PATH', 'data/processed'))
    csv_path = csv_path / os.getenv('DATASET_NAME', "nhanes.csv")
    
    load_nhanes_survival(csv_path=csv_path)