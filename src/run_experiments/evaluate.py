from typing import Optional, Sequence

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    integrated_brier_score,
    brier_score,
    cumulative_dynamic_auc
)

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


def _compute_nri(
    event_probs_new: np.ndarray,
    event_probs_baseline: np.ndarray,
    events: np.ndarray,
    times: np.ndarray,
    eval_time: float,
    risk_thresholds: tuple = (0.1, 0.2)
) -> dict:
    """
    Compute Net Reclassification Improvement (NRI) comparing new model to baseline.
    
    Categorizes patients into risk groups and calculates how many are correctly reclassified.
    
    Parameters:
    -----------
    event_probs_new : array of predicted event probabilities from new model
    event_probs_baseline : array of predicted event probabilities from baseline model
    events : boolean array indicating if event occurred
    times : array of event/censoring times
    eval_time : time point for evaluation
    risk_thresholds : tuple of (low_threshold, high_threshold) defining risk categories
    
    Returns:
    --------
    dict with NRI metrics
    """
    low_thresh, high_thresh = risk_thresholds
    
    # Only consider patients with sufficient follow-up or who had event before eval_time
    valid_mask = (times >= eval_time) | (events & (times < eval_time))
    
    if not valid_mask.any():
        return {
            "nri": float("nan"),
            "nri_events": float("nan"),
            "nri_nonevents": float("nan"),
            "n_valid": 0
        }
    
    # Actual outcomes at eval_time
    had_event = events & (times <= eval_time)
    
    # Filter to valid patients
    event_probs_new_valid = event_probs_new[valid_mask]
    event_probs_baseline_valid = event_probs_baseline[valid_mask]
    had_event_valid = had_event[valid_mask]
    
    # Categorize into risk groups: 0=low, 1=medium, 2=high
    def categorize_risk(probs, low, high):
        categories = np.zeros(len(probs), dtype=int)
        categories[probs >= high] = 2
        categories[(probs >= low) & (probs < high)] = 1
        return categories
    
    risk_cat_new = categorize_risk(event_probs_new_valid, low_thresh, high_thresh)
    risk_cat_baseline = categorize_risk(event_probs_baseline_valid, low_thresh, high_thresh)
    
    # NRI for patients with events (correctly moved up - incorrectly moved down)
    events_mask = had_event_valid
    if events_mask.sum() > 0:
        moved_up_events = (risk_cat_new > risk_cat_baseline)[events_mask].sum()
        moved_down_events = (risk_cat_new < risk_cat_baseline)[events_mask].sum()
        nri_events = (moved_up_events - moved_down_events) / events_mask.sum()
    else:
        nri_events = float("nan")
    
    # NRI for patients without events (correctly moved down - incorrectly moved up)
    nonevents_mask = ~had_event_valid
    if nonevents_mask.sum() > 0:
        moved_down_nonevents = (risk_cat_new < risk_cat_baseline)[nonevents_mask].sum()
        moved_up_nonevents = (risk_cat_new > risk_cat_baseline)[nonevents_mask].sum()
        nri_nonevents = (moved_down_nonevents - moved_up_nonevents) / nonevents_mask.sum()
    else:
        nri_nonevents = float("nan")
    
    # Total NRI
    if not np.isnan(nri_events) and not np.isnan(nri_nonevents):
        nri = nri_events + nri_nonevents
    else:
        nri = float("nan")
    
    return {
        "nri": float(nri),
        "nri_events": float(nri_events),
        "nri_nonevents": float(nri_nonevents),
        "n_valid": int(valid_mask.sum()),
        "n_events": int(events_mask.sum()),
        "n_nonevents": int(nonevents_mask.sum())
    }


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

    train_cindex_harrell, train_n_conc, train_n_disc, train_n_tied_risk, train_n_tied_time = concordance_index_censored(
        event_indicator=event_train,
        event_time=time_train,
        estimate=train_risk,
    )
    
    artifacts.append_metric({"name": "train_cindex", "data": float(train_cindex_harrell)})
    artifacts.append_data({"name": "train_cindex_concordant_pairs", "data": int(train_n_conc)})
    artifacts.append_data({"name": "train_cindex_discordant_pairs", "data": int(train_n_disc)})
    artifacts.append_data({"name": "train_cindex_tied_risk_pairs", "data": int(train_n_tied_risk)})
    artifacts.append_data({"name": "train_cindex_tied_time_pairs", "data": int(train_n_tied_time)})

    # Harrell's C-index
    cindex_harrell, n_conc, n_disc, n_tied_risk, n_tied_time = concordance_index_censored(
        event_indicator=event_test,
        event_time=time_test,
        estimate=test_risk,
    )

    artifacts.append_metric({"name": "test_cindex_harrell", "data": float(cindex_harrell)})
    artifacts.append_metric({"name": "test_cindex_harrell_concordant_pairs", "data": int(n_conc)})
    artifacts.append_metric({"name": "test_cindex_harrell_discordant_pairs", "data": int(n_disc)})
    artifacts.append_metric({"name": "test_cindex_harrell_tied_risk_pairs", "data": int(n_tied_risk)})
    artifacts.append_metric({"name": "test_cindex_harrell_tied_time_pairs", "data": int(n_tied_time)})
    
    # Uno's C-index (more robust to censoring)
    try:
        train_cindex_uno, train_concordant_uno, train_discordant_uno, train_tied_risk_uno, train_tied_time_uno = concordance_index_ipcw(
            survival_train=train_y,
            survival_test=train_y,
            estimate=train_risk,
        )
        artifacts.append_metric({"name": "train_cindex_uno", "data": float(train_cindex_uno)})
        artifacts.append_metric({"name": "train_cindex_uno_concordant_pairs", "data": int(train_concordant_uno)})
        artifacts.append_metric({"name": "train_cindex_uno_discordant_pairs", "data": int(train_discordant_uno)})
        artifacts.append_metric({"name": "train_cindex_uno_tied_risk_pairs", "data": int(train_tied_risk_uno)})
        artifacts.append_metric({"name": "train_cindex_uno_tied_time_pairs", "data": int(train_tied_time_uno)})

        cindex_uno, concordant_uno, discordant_uno, tied_risk_uno, tied_time_uno = concordance_index_ipcw(
            survival_train=train_y,
            survival_test=test_y,
            estimate=test_risk,
        )
        artifacts.append_metric({"name": "test_cindex_uno", "data": float(cindex_uno)})
        artifacts.append_metric({"name": "test_cindex_uno_concordant_pairs", "data": int(concordant_uno)})
        artifacts.append_metric({"name": "test_cindex_uno_discordant_pairs", "data": int(discordant_uno)})
        artifacts.append_metric({"name": "test_cindex_uno_tied_risk_pairs", "data": int(tied_risk_uno)})
        artifacts.append_metric({"name": "test_cindex_uno_tied_time_pairs", "data": int(tied_time_uno)})
    except Exception as e:
        if verbose:
            print(f"Warning: Could not compute Uno's C-index (heavy censoring). Using Harrell's C-index only.")
        # Store NaN to indicate metric couldn't be computed
        artifacts.append_metric({"name": "test_cindex_uno", "data": float("nan")})

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

        # Time-specific Brier scores at key time points
        try:
            # Define evaluation time points (in months for your data)
            eval_times_brier = np.array([12.0, 24.0, 36.0, 60.0, 120.0, 180.0])  # 1, 2, 3, 5, 10, 15 years
            # Filter to times within the observed range
            eval_times_brier = eval_times_brier[(eval_times_brier >= t_min) & (eval_times_brier <= t_max)]
            
            if len(eval_times_brier) > 0:
                # Generate survival matrix at these specific time points
                _, surv_at_eval_times = _predict_survival_matrix(model, test_X, time_grid=eval_times_brier)
                
                # Compute Brier score - note: returns (times, scores) not (scores, times)
                brier_times, brier_scores = brier_score(
                    survival_train=train_y,
                    survival_test=test_y,
                    estimate=surv_at_eval_times,
                    times=eval_times_brier
                )
                
                # Store individual time-specific Brier scores
                for eval_time, bs_value in zip(brier_times, brier_scores):
                    artifacts.append_metric({
                        "name": f"test_brier_score_at_{int(eval_time)}_months",
                        "data": float(bs_value)
                    })
                
                artifacts.append_data({"name": "time_specific_brier_times", "data": brier_times})
                artifacts.append_data({"name": "time_specific_brier_scores", "data": brier_scores})
        except Exception as e:
            if verbose:
                print(f"Could not compute time-specific Brier scores: {e}")

        # Time-dependent AUC at specific time points
        try:
            # Define evaluation time points (in months for your data)
            eval_times = np.array([12, 24, 36, 60, 120, 180])  # 1, 2, 3, 5, 10, 15 years
            # Filter to times within the observed range and with sufficient follow-up
            # Only use times where we have events (not just censoring)
            max_event_time = float(np.max(time_test[event_test]))
            eval_times = eval_times[(eval_times >= t_min) & (eval_times <= min(t_max, max_event_time * 0.9))]
            
            if len(eval_times) > 0:
                auc_scores, mean_auc = cumulative_dynamic_auc(
                    survival_train=train_y,
                    survival_test=test_y,
                    estimate=test_risk,
                    times=eval_times
                )
                
                # Store time-specific AUC scores
                time_specific_aucs = {
                    f"test_auc_at_{int(t)}_months": float(auc)
                    for t, auc in zip(eval_times, auc_scores)
                }
                for name, value in time_specific_aucs.items():
                    artifacts.append_metric({"name": name, "data": value})
                
                artifacts.append_metric({"name": "test_mean_dynamic_auc", "data": float(mean_auc)})
                artifacts.append_data({"name": "time_dependent_auc_times", "data": eval_times})
                artifacts.append_data({"name": "time_dependent_auc_scores", "data": auc_scores})
            elif verbose:
                print(f"Warning: No valid time points for AUC computation (max event time: {max_event_time:.1f} months)")
        except Exception as e:
            if verbose:
                print(f"Warning: Could not compute time-dependent AUC (insufficient events at evaluation times).")
        
        # Net Reclassification Improvement (NRI) - compare to baseline risk predictions
        # Store baseline predictions if this is a baseline experiment
        experiment_name = experiment.name if experiment else "unknown"
        if experiment and experiment.metadata and "dataset" in experiment.metadata:
            dataset_name = experiment.metadata["dataset"]
            
            # Check if we should compute NRI (not for baseline experiments)
            is_baseline = "bmi-only" in experiment_name or "waist-to-height-only" in experiment_name
            
            # Try to load baseline predictions for NRI comparison
            if not is_baseline:
                try:
                    from ..common import get_experiment_artifacts_path
                    
                    model_type = experiment_name.split('_')[0]  # e.g., 'cox', 'random-forest', 'svm'
                    
                    # Compare against both BMI-only and waist-to-height-only baselines
                    baseline_configs = [
                        ("bmi", f"{model_type}_{dataset_name}_bmi-only"),
                        ("waist-to-height", f"{model_type}_{dataset_name}_waist-to-height-only")
                    ]
                    
                    for baseline_label, baseline_name in baseline_configs:
                        try:
                            baseline_path = get_experiment_artifacts_path(baseline_name)
                            
                            if baseline_path.exists():
                                from ..common import ExperimentArtifacts as EA
                                baseline_artifacts = EA.load(baseline_path)
                                
                                if hasattr(baseline_artifacts, 'model') and baseline_artifacts.model is not None:
                                    # Compute event probabilities at evaluation times for NRI
                                    eval_times_nri = np.array([60.0, 120.0])  # 5 and 10 years
                                    eval_times_nri = eval_times_nri[(eval_times_nri >= t_min) & (eval_times_nri <= t_max)]
                                    
                                    if len(eval_times_nri) > 0:
                                        # Get predictions from both models
                                        # Important: use baseline's preprocessor for baseline predictions
                                        _, surv_new_nri = _predict_survival_matrix(model, test_X, time_grid=eval_times_nri)
                                        
                                        # Transform test data using baseline's preprocessor
                                        test_X_baseline = baseline_artifacts.preprocessor.transform(test_X_raw)
                                        _, surv_baseline_nri = _predict_survival_matrix(baseline_artifacts.model, test_X_baseline, time_grid=eval_times_nri)
                                        
                                        # Compute NRI at each time point
                                        for i, eval_time in enumerate(eval_times_nri):
                                            event_probs_new = 1.0 - surv_new_nri[:, i]
                                            event_probs_baseline = 1.0 - surv_baseline_nri[:, i]
                                            
                                            nri_results = _compute_nri(
                                                event_probs_new=event_probs_new,
                                                event_probs_baseline=event_probs_baseline,
                                                events=event_test,
                                                times=time_test,
                                                eval_time=eval_time,
                                                risk_thresholds=(0.1, 0.2)  # 10% and 20% risk thresholds
                                            )
                                            
                                            # Store NRI metrics with baseline-specific naming
                                            time_label = int(eval_time)
                                            artifacts.append_metric({
                                                "name": f"test_nri_vs_{baseline_label}_at_{time_label}_months",
                                                "data": nri_results["nri"]
                                            })
                                            artifacts.append_metric({
                                                "name": f"test_nri_events_vs_{baseline_label}_at_{time_label}_months",
                                                "data": nri_results["nri_events"]
                                            })
                                            artifacts.append_metric({
                                                "name": f"test_nri_nonevents_vs_{baseline_label}_at_{time_label}_months",
                                                "data": nri_results["nri_nonevents"]
                                            })
                        except Exception as e:
                            if verbose:
                                print(f"Note: Could not compute NRI vs {baseline_label}: {e}")

                except Exception as e:
                    if verbose:
                        print(f"Note: Could not compute NRI (baseline models not available): {e}")

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
