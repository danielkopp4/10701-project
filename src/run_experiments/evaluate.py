from .constructs import Experiment, ExperimentArtifacts
from .synthetic_data_generator import simulate_I_ground_truth_counterfactuals
from sklearn.metrics import accuracy_score
import numpy as np
from sksurv.metrics import (
    concordance_index_censored,
    integrated_brier_score,
)

def _extract_event_time(y):
    """
    y: structured array (e.g. sksurv.util.Surv) with fields 'event' and 'time'.
    Returns (event_indicator, event_time) as 1D numpy arrays.
    """
    # If using sksurv Surv, the field names are exactly 'event' and 'time'.
    event = np.asarray(y["event"], dtype=bool)
    time = np.asarray(y["time"], dtype=float)
    return event, time


def _predict_survival_matrix(model, X, time_grid=None):
    """
    Convert model.predict_survival_function(X) into:
      - times: 1D array of time points
      - surv: 2D array of survival probabilities (n_samples, n_times)

    Works with scikit-survival estimators that return StepFunction objects.
    """
    if not hasattr(model, "predict_survival_function"):
        return None, None

    surv_funcs = model.predict_survival_function(X)

    # If caller didn’t provide a time grid, build one from all functions
    if time_grid is None:
        all_times = np.unique(
            np.concatenate([fn.x for fn in surv_funcs])
        )
        time_grid = all_times

    # Evaluate each survival function on the common grid
    surv_matrix = np.row_stack([fn(time_grid) for fn in surv_funcs])

    return time_grid, surv_matrix

def _compute_individual_index(
    model,
    X,
    time_grid,
    t_eval,
    A_name: str,
    A_values=None,
):
    if A_values is None:
        A_values = np.sort(X[A_name].unique())

    A_values = list(A_values)
    n_samples = X.shape[0]
    n_levels = len(A_values)

    rows = []
    for i in range(n_samples):
        x_i = X.iloc[i]
        for a in A_values:
            x_cf = x_i.copy()
            x_cf[A_name] = a
            rows.append(x_cf)

    X_cf = X.__class__(rows)

    surv_cf_list = model.predict_survival_function(X_cf)

    t_idx = int(np.argmin(np.abs(time_grid - t_eval)))
    t_eval_actual = float(time_grid[t_idx])

    event_probs_cf = np.array([
        1.0 - sf(time_grid)[t_idx]
        for sf in surv_cf_list
    ])  # shape (n_samples * n_levels,)
    event_probs_cf = event_probs_cf.reshape(n_samples, n_levels)

    min_test_P_i = event_probs_cf.min(axis=1)

    A_to_idx = {a: j for j, a in enumerate(A_values)}
    a_obs = X[A_name]
    obs_idx = np.array([A_to_idx[a] for a in a_obs])

    p_obs = event_probs_cf[np.arange(n_samples), obs_idx]

    pred_index = p_obs - min_test_P_i

    return pred_index, min_test_P_i, p_obs, A_values, t_eval_actual


def evaluate_experiment(data, 
                        experiment: Experiment, 
                        sensitive_attr: str = "weight",   # name of A in X
                        sensitive_levels=None,
                        verbose=True, 
                        test_causal=True
                    ) -> ExperimentArtifacts:
    """
    Given some model and training procedure, run evaluation criteria on it to see
    how well it fares.
    Data can either be purely real, purely synthetic, or a mix of both.

    Evaluation criteria:
    - intermediate criteria on prediction quality for real data + synthetic
    - internal consistency on synthetic benchmarks

    From this we will generate artifacts that can be used during analysis.
    All results from evaluation will be stored in addition to the trained models
    for the purpose of generating plots.
    """
    # one factor of evaluation is to see regression performance on real data
    # another is to see how well it captures known structure in synthetic data
    # the final factor is to see how well it captures causal factors in synthetic data
    artifacts = ExperimentArtifacts()
    
    # split the train and test data
    train_X, train_y, test_X, test_y = data
    artifacts.append_data({"name": "dataset", "data": data})
    
    # train the model
    model = experiment.model
    model.train(train_X, train_y, test_X, test_y)
    artifacts.set_model(model)
    
    ## Evaluate IID ##
    
    # generate risk scores - any 1D ranking of individuals (higher = worse)
    test_risk = model.predict(test_X)
    train_risk = model.predict(train_X)
    artifacts.append_data({"name": "train_risk_scores", "data": train_risk})
    artifacts.append_data({"name": "test_risk_scores", "data": test_risk})

    event_test, time_test = _extract_event_time(test_y)

    # Harrell's C-index (rank-based performance)
    cindex, n_conc, n_disc, n_tied_risk, n_tied_time = concordance_index_censored(
        event_indicator=event_test,
        event_time=time_test,
        estimate=test_risk,
    )

    artifacts.append_metric({"name": "test_cindex", "data": float(cindex)})
    artifacts.append_metric({"name": "test_cindex_concordant_pairs", "data": int(n_conc)})
    artifacts.append_metric({"name": "test_cindex_discordant_pairs", "data": int(n_disc)})
    artifacts.append_metric({"name": "test_cindex_tied_risk_pairs", "data": int(n_tied_risk)})
    artifacts.append_metric({"name": "test_cindex_tied_time_pairs", "data": int(n_tied_time)})

    # survival curves - only available if model can predict survival functions
    time_grid = None
    surv_test = None
    surv_train = None

    if hasattr(model, "predict_survival_function"):
        _, time_train = _extract_event_time(train_y)
        t_min = float(np.min(time_train))
        t_max = float(np.max(time_train))

        time_grid = np.linspace(t_min, t_max, 50)
        time_grid, surv_test = _predict_survival_matrix(model, test_X, time_grid=time_grid)
        _, surv_train = _predict_survival_matrix(model, train_X, time_grid=time_grid)

        artifacts.append_data({"name": "time_grid", "data": time_grid})
        artifacts.append_data({"name": "test_survival_probabilities", "data": surv_test})
        artifacts.append_data({"name": "train_survival_probabilities", "data": surv_train})

        ibs = integrated_brier_score(
            survival_train=train_y,
            survival_test=test_y,
            estimate=surv_test,
            times=time_grid,
        )
        artifacts.append_metric({"name": "test_integrated_brier_score", "data": float(ibs)})

    ## Compute Indices ##
    pred_index = None

    if surv_test is not None and time_grid is not None:
        t_eval = experiment.T_eval # float(np.median(time_test))
        
        pred_index, min_test_P_i, p_obs, A_values, t_eval_actual = _compute_individual_index(
            model=model,
            X=test_X,
            time_grid=time_grid,
            t_eval=t_eval,
            A_name=sensitive_attr,
            A_values=sensitive_levels,
        )

        pred_pop_index = float(np.mean(pred_index))

        artifacts.append_data({
            "name": "predicted_risk_index",
            "data": pred_index,
        })
        artifacts.append_data({
            "name": "predicted_min_event_prob_per_individual",
            "data": min_test_P_i,
        })
        artifacts.append_data({
            "name": "predicted_event_prob_at_eval_time_obs_A",
            "data": {
                "t_eval_actual": t_eval_actual,
                "t_eval": t_eval,
                "probs": p_obs,
                "attr_name": sensitive_attr,
                "attr_values": A_values,
            },
        })
        artifacts.append_data({
            "name": "predicted_risk_pop_index",
            "data": pred_pop_index,
        })
    
    ## Evaluate Causal ##
    if test_causal and pred_index is not None:
        gt_index = simulate_I_ground_truth_counterfactuals(test_X)
        gt_index = np.asarray(gt_index, dtype=float)

        gt_pop_index = float(np.mean(gt_index))

        artifacts.append_data({"name": "gt_risk_index", "data": gt_index})
        artifacts.append_data({"name": "gt_risk_pop_index", "data": gt_pop_index})

        # Metrics comparing predicted vs. ground-truth index
        # Pearson correlation
        if pred_index.shape == gt_index.shape:
            corr = float(np.corrcoef(pred_index, gt_index)[0, 1])
        else:
            print("ERROR: shape mismatch when computing correlation between predicted and ground-truth risk index.")
            return None

        pop_index_diff = float(np.abs(pred_pop_index - gt_pop_index))
        rmse = float(np.sqrt(np.mean((pred_index - gt_index) ** 2)))

        artifacts.append_metric({"name": "pred_vs_gt_index_corr", "data": corr})
        artifacts.append_metric({"name": "pred_vs_gt_pop_index_diff", "data": pop_index_diff})
        artifacts.append_metric({"name": "pred_vs_gt_index_rmse", "data": rmse})
    
    if verbose:
        artifacts.print_summary()
    
    return artifacts
