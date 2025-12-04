from typing import Optional, Sequence

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from ..common import Experiment, ExperimentArtifacts, Preprocessor
from .synthetic_data.simulator import simulate_I_ground_truth_counterfactuals


def _extract_event_time(y):
    # y is sksurv Surv array with fields 'event' and 'time'
    event = np.asarray(y["event"], dtype=bool)
    time = np.asarray(y["time"], dtype=float)
    return event, time


def _predict_survival_matrix(model, X_proc, time_grid=None):
    """
    Convert model.predict_survival_function(X_proc) into:
      - times: 1D array of time points
      - surv: 2D array of survival probabilities (n_samples, n_times)
    """
    if not hasattr(model, "predict_survival_function"):
        return None, None

    surv_funcs = model.predict_survival_function(X_proc)

    if time_grid is None:
        all_times = np.unique(
            np.concatenate([fn.x for fn in surv_funcs])
        )
        time_grid = all_times

    surv_matrix = np.vstack([fn(time_grid) for fn in surv_funcs])
    return time_grid, surv_matrix

def _fit_isotonic_calibrator_at_T(
    time_grid: np.ndarray,
    surv_train: np.ndarray,
    y_train,
    T_eval: float,
):
    """
    Fit an isotonic regression calibrator for p(event by T_eval).
    """
    event_train, time_train = _extract_event_time(y_train)

    # choose the grid index closest to T_eval
    t_idx = int(np.argmin(np.abs(time_grid - T_eval)))
    t_eval_actual = float(time_grid[t_idx])

    # model-based probability of event by t_eval_actual
    p_hat = 1.0 - surv_train[:, t_idx]

    # observed indicator of event by t_eval_actual
    Y = ((event_train == 1) & (time_train <= t_eval_actual)).astype(float)

    # if there is no variation, isotonic regression is not identifiable
    if np.all(Y == Y[0]):
        # no information to calibrate
        return None, t_eval_actual

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_hat, Y)

    return iso, t_eval_actual



def _compute_individual_index(
    model,
    preprocessor,
    X_raw,
    time_grid,
    t_eval,
    A_name: str,
    A_values: Optional[Sequence] = None,
    A_strata: int = 300,
    w_bounds: Optional[tuple[float, float]] = None,
    softmin_q: Optional[float] = 0.05,
    calibrator=None,  # can be an IsotonicRegression or any callable p -> p_cal
):
    """
    Compute prediction-based individual index

        I_i(T) = p_i(A_obs, T) - min_a p_i(a, T)
    """

    # Ensure time_grid is a proper 1D array
    time_grid = np.asarray(time_grid, dtype=float).ravel()
    if time_grid.ndim != 1 or time_grid.size == 0:
        raise ValueError("time_grid must be a non-empty 1D array")

    # Make sure A feature wasn't removed by the preprocessor
    if getattr(preprocessor, "feature_cols_", None) is not None:
        if A_name not in preprocessor.feature_cols_:
            raise ValueError(
                f"_compute_individual_index: preprocessor removed A={A_name} feature"
            )

    a_obs = X_raw[A_name]

    a_nonmissing = a_obs.dropna().to_numpy()
    if A_values is not None:
        A_values = np.array(sorted(set(A_values)), dtype=float)
        if len(A_values) == 0:
            raise ValueError(
                f"_compute_individual_index: provided A_values for {A_name} is empty"
            )
    else:
        if w_bounds is not None:
            lo, hi = w_bounds
            n_eff = min(A_strata, 50)  # cap resolution for stability
            A_values = np.linspace(lo, hi, num=n_eff)
        else:
            n_eff = max(2, min(A_strata, len(a_nonmissing)))
            quantiles = np.linspace(0.0, 1.0, n_eff)
            edges = np.unique(np.quantile(a_nonmissing, quantiles))
            if len(edges) < 2:
                raise ValueError(
                    f"_compute_individual_index: not enough non-missing values for {A_name}"
                )
            mids = 0.5 * (edges[:-1] + edges[1:])
            A_values = np.unique(mids.astype(float))

    A_values = np.asarray(A_values, dtype=float)
    n_samples = X_raw.shape[0]
    n_levels = len(A_values)

    # Build counterfactual panel
    rows = []
    for i in range(n_samples):
        x_i = X_raw.iloc[i]
        for a in A_values:
            x_cf = x_i.copy()
            x_cf[A_name] = a
            rows.append(x_cf)

    X_cf_raw = X_raw.__class__(rows)
    X_cf_proc = preprocessor.transform(X_cf_raw)

    surv_cf_list = model.predict_survival_function(X_cf_proc)

    # common evaluation time index
    t_idx = int(np.argmin(np.abs(time_grid - t_eval)))
    t_eval_actual = float(time_grid[t_idx])

    # counterfactual event probabilities p_i(a, T_eval)
    event_probs_cf = np.array([
        1.0 - np.atleast_1d(sf(time_grid))[t_idx]
        for sf in surv_cf_list
    ])  # shape (n_samples * n_levels,)
    event_probs_cf = event_probs_cf.reshape(n_samples, n_levels)

    # optional calibration of p_i(a, T_eval)
    if calibrator is not None:
        flat_cf = event_probs_cf.reshape(-1)
        flat_cf = np.clip(flat_cf, 0.0, 1.0)
        flat_cf_cal = calibrator.transform(flat_cf)
        event_probs_cf = flat_cf_cal.reshape(event_probs_cf.shape)

    # min_a p_i(a, T_eval): strict min or "softmin" via quantile
    if softmin_q is None:
        min_test_P_i = event_probs_cf.min(axis=1)
    else:
        q = float(softmin_q)
        min_test_P_i = np.quantile(event_probs_cf, q, axis=1)

    # observed path: p_i(A_obs, T_eval)
    X_obs_proc = preprocessor.transform(X_raw)
    surv_obs_list = model.predict_survival_function(X_obs_proc)
    event_probs_obs = np.array([
        1.0 - np.atleast_1d(sf(time_grid))[t_idx]
        for sf in surv_obs_list
    ])

    if calibrator is not None:
        p_obs_clipped = np.clip(event_probs_obs, 0.0, 1.0)
        event_probs_obs = calibrator.transform(p_obs_clipped)

    valid_mask = a_obs.notna()
    if not valid_mask.any():
        raise ValueError(
            f"_compute_individual_index: all values of A={A_name} are missing for this split"
        )

    # p_i(A_obs, T_eval), NaN where A is missing
    p_obs = np.full(n_samples, np.nan, dtype=float)
    p_obs[valid_mask] = event_probs_obs[valid_mask]

    # I_i(T) = p_obs - min_a p_i(a, T_eval), NaN where p_obs is NaN
    pred_index = np.full(n_samples, np.nan, dtype=float)
    pred_index[valid_mask] = p_obs[valid_mask] - min_test_P_i[valid_mask]

    return pred_index, min_test_P_i, p_obs, A_values, t_eval_actual


def evaluate_experiment(
    data,
    experiment: Experiment,
    target_feature: str = "BMXWT",   # name of A in raw X
    verbose: bool = True,
    test_causal: bool = True,
) -> ExperimentArtifacts:
    artifacts = ExperimentArtifacts(experiment=experiment)

    # raw data from loader
    train_X_raw, train_y, test_X_raw, test_y = data
    # artifacts.append_data({"name": "dataset_raw", "data": data})

    # preprocessing
    preproc: Preprocessor = experiment.preprocessor
    preproc.fit(train_X_raw, train_y)

    train_X = preproc.transform(train_X_raw)
    test_X = preproc.transform(test_X_raw)

    artifacts.set_preprocessor(preproc)
    artifacts.append_data({"name": "feature_names", "data": getattr(preproc, "feature_cols_", None)})

    # train model
    model = experiment.model
    model.train(train_X, train_y, test_X, test_y)
    artifacts.set_model(model)

    # IID evaluation
    test_risk = model.predict(test_X)
    train_risk = model.predict(train_X)
    artifacts.append_data({"name": "train_risk_scores", "data": train_risk})
    artifacts.append_data({"name": "test_risk_scores", "data": test_risk})

    event_train, time_train = _extract_event_time(train_y)
    event_test, time_test = _extract_event_time(test_y)

    train_cindex, train_n_conc, train_n_disc, train_n_tied_risk, train_n_tied_time = concordance_index_censored(
        event_indicator=event_train,
        event_time=time_train,
        estimate=train_risk,
    )
    
    artifacts.append_metric({"name": "train_cindex", "data": float(train_cindex)})
    artifacts.append_data({"name": "train_cindex_concordant_pairs", "data": int(train_n_conc)})
    artifacts.append_data({"name": "train_cindex_discordant_pairs", "data": int(train_n_disc)})
    artifacts.append_data({"name": "train_cindex_tied_risk_pairs", "data": int(train_n_tied_risk)})
    artifacts.append_data({"name": "train_cindex_tied_time_pairs", "data": int(train_n_tied_time)})

    test_cindex, test_n_conc, test_n_disc, test_n_tied_risk, test_n_tied_time = concordance_index_censored(
        event_indicator=event_test,
        event_time=time_test,
        estimate=test_risk,
    )

    artifacts.append_metric({"name": "test_cindex", "data": float(test_cindex)})
    artifacts.append_data({"name": "test_cindex_concordant_pairs", "data": int(test_n_conc)})
    artifacts.append_data({"name": "test_cindex_discordant_pairs", "data": int(test_n_disc)})
    artifacts.append_data({"name": "test_cindex_tied_risk_pairs", "data": int(test_n_tied_risk)})
    artifacts.append_data({"name": "test_cindex_tied_time_pairs", "data": int(test_n_tied_time)})

    # survival curves (processed X)
    time_grid = None
    surv_test = None
    surv_train = None
    calibrator = None

    if hasattr(model, "predict_survival_function"):
        event_test, time_test = _extract_event_time(test_y)

        t_min = float(np.min(time_test))
        t_max = float(np.max(time_test))

        # grid for standard survival metrics / curves
        metrics_time_grid = np.linspace(t_min, t_max, 50, endpoint=False)
        metrics_time_grid = np.unique(np.append(metrics_time_grid, float(experiment.T_eval)))

        metrics_time_grid, surv_test = _predict_survival_matrix(model, test_X, time_grid=metrics_time_grid)
        _, surv_train = _predict_survival_matrix(model, train_X, time_grid=metrics_time_grid)
        time_grid = metrics_time_grid  # downstream gating uses this variable

        artifacts.append_data({"name": "time_grid", "data": metrics_time_grid})
        artifacts.append_data({"name": "test_survival_probabilities", "data": surv_test})
        artifacts.append_data({"name": "train_survival_probabilities", "data": surv_train})

        ibs = integrated_brier_score(
            survival_train=train_y,
            survival_test=test_y,
            estimate=surv_test,
            times=metrics_time_grid,
        )
        artifacts.append_metric({"name": "test_integrated_brier_score", "data": float(ibs)})

        # No calibration for causal eval (empirically improves index agreement)
        calibrator = None


    # estimated index
    pred_index = None
    pred_pop_index = None

    if test_causal and surv_test is not None and time_grid is not None:
            t_eval = float(experiment.T_eval)
            T_eval_int = int(round(experiment.T_eval))
            time_grid_causal = np.array([t_eval], dtype=float)

            # fixed weight bounds for oracle alignment
            w_bounds = (10.0, 200.0)

            pred_index, min_test_P_i, p_obs, A_values, t_eval_actual = _compute_individual_index(
                model=model,
                preprocessor=preproc,
                X_raw=test_X_raw,
                time_grid=time_grid_causal,
                t_eval=t_eval,
                A_name=target_feature,
                A_strata=experiment.A_strata,
                w_bounds=w_bounds,
                softmin_q=None,  # hard min to match oracle
                calibrator=calibrator
            )

            pred_pop_index = float(np.nanmean(pred_index))

            artifacts.append_data({"name": "predicted_risk_index", "data": pred_index})
            artifacts.append_data({"name": "predicted_min_event_prob_per_individual", "data": min_test_P_i})
            artifacts.append_data({
                "name": "predicted_event_prob_at_eval_time_obs_A",
                "data": {
                    "t_eval_actual": t_eval_actual,
                    "t_eval": t_eval,
                    "probs": p_obs,
                    "attr_name": target_feature,
                    "attr_values": A_values,
                },
            })
            artifacts.append_data({"name": "predicted_risk_pop_index", "data": pred_pop_index})

            # causal evaluation vs simulator oracle
            T_eval_int = int(round(experiment.T_eval))

            I_df = simulate_I_ground_truth_counterfactuals(
                test_X_raw, T_eval=T_eval_int, w_bounds=w_bounds
            )
            gt_index = np.asarray(I_df["I_i"], dtype=float)

            gt_pop_index = float(np.nanmean(gt_index))

            artifacts.append_data({"name": "gt_risk_index", "data": gt_index})
            artifacts.append_data({"name": "gt_risk_pop_index", "data": gt_pop_index})

            if pred_index.shape != gt_index.shape:
                print(
                    "ERROR: shape mismatch between predicted and ground-truth "
                    "risk index; not computing correlation."
                )
                return artifacts

            common_mask = ~np.isnan(pred_index) & ~np.isnan(gt_index)
            n_common = int(common_mask.sum())

            if n_common == 0:
                print(
                    "WARNING: no overlapping non-NaN entries for pred vs gt index; "
                    "causal comparison metrics set to NaN."
                )
                corr = float("nan")
                pop_index_diff = float("nan")
                rmse = float("nan")
            else:
                if np.all(pred_index[common_mask] == pred_index[common_mask][0]):
                    print("WARNING: predicted risk index is constant across all individuals;")
                    corr = float("nan")
                else:
                    corr = float(np.corrcoef(pred_index[common_mask], gt_index[common_mask])[0, 1])

                pop_index_diff = float(
                    np.nanmean(pred_index) - np.nanmean(gt_index)
                )
                rmse = float(
                    np.sqrt(
                        np.mean((pred_index[common_mask] - gt_index[common_mask]) ** 2)
                    )
                )

            artifacts.append_metric({"name": "pred_vs_gt_index_corr", "data": corr})
            artifacts.append_metric({"name": "pred_vs_gt_pop_index_diff", "data": pop_index_diff})
            artifacts.append_metric({"name": "pred_vs_gt_index_rmse", "data": rmse})


    if verbose:
        artifacts.print_summary()

    return artifacts
