from typing import Dict

from .constructs import Experiment, ExperimentArtifacts
from .models import Cox
from .preprocessors import NHANESPreprocessor
from .evaluate import evaluate_experiment
from ..get_data.dataset_api import load_nhanes_survival  # your real-data loader
from .synthetic_data.simulator import (
    load_nhanes_survival_simulated
)


def run_experiments(
    nhanes_csv_path: str,
    synthetic_n: int = 10000,
    T_eval_years: float = 10.0,
    sensitive_attr: str = "BMXWT",
    verbose: bool = True,
) -> Dict[str, ExperimentArtifacts]:
    T_eval_months = T_eval_years * 12.0
    artifacts_by_name: Dict[str, ExperimentArtifacts] = {}

    real_data = load_nhanes_survival(csv_path=nhanes_csv_path)
    synth_data = load_nhanes_survival_simulated(
        n=synthetic_n
    )
    
    
    # real NHANES
    real_preproc = NHANESPreprocessor(
        exclude_downstream=True,
        impute_strategy="median",
    )
    real_model = Cox(alpha=0.1) # alpha to avioid overflow

    exp_real = Experiment(
        name="cox_real_nhanes",
        model=real_model,
        preprocessor=real_preproc,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_real"},
    )

    art_real = evaluate_experiment(
        data=real_data,
        experiment=exp_real,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=False,  # no simulator oracle for real data
    )
    artifacts_by_name[exp_real.name] = art_real

    # synthetic NHANES
    synth_preproc = NHANESPreprocessor(
        exclude_downstream=True,
        impute_strategy="median",
    )
    synth_model = Cox(alpha=0.1)  # alpha to avioid overflow

    exp_synth = Experiment(
        name="cox_synthetic_nhanes",
        model=synth_model,
        preprocessor=synth_preproc,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_synthetic"},
    )

    art_synth = evaluate_experiment(
        data=synth_data,
        experiment=exp_synth,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=True
    )
    artifacts_by_name[exp_synth.name] = art_synth
    
    # ---------------------------------------------------------
    
    # real NHANES
    real_preproc_2 = NHANESPreprocessor(
        exclude_all=True,
        impute_strategy="median",
    )
    real_model_2 = Cox(alpha=0.1) # alpha to avioid overflow

    exp_real_2 = Experiment(
        name="cox_real_nhanes_classic",
        model=real_model_2,
        preprocessor=real_preproc_2,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_real"},
    )

    art_real = evaluate_experiment(
        data=real_data,
        experiment=exp_real_2,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=False,  # no simulator oracle for real data
    )
    artifacts_by_name[exp_real_2.name] = art_real

    # synthetic NHANES
    synth_preproc_2 = NHANESPreprocessor(
        exclude_all=True,
        impute_strategy="median",
    )
    synth_model_2 = Cox(alpha=0.1)  # alpha to avioid overflow

    exp_synth_2 = Experiment(
        name="cox_synthetic_nhanes_classic",
        model=synth_model_2,
        preprocessor=synth_preproc_2,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_synthetic"},
    )

    art_synth = evaluate_experiment(
        data=synth_data,
        experiment=exp_synth_2,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=True
    )
    artifacts_by_name[exp_synth_2.name] = art_synth
    
    # ---------------------------------------------------------
    
    real_preproc_3 = NHANESPreprocessor(
        exclude_downstream=False,
        impute_strategy="median",
    )
    real_model_3 = Cox(alpha=1) # alpha to avioid overflow

    exp_real_3 = Experiment(
        name="cox_real_nhanes_naive",
        model=real_model_3,
        preprocessor=real_preproc_3,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_real"},
    )

    art_real = evaluate_experiment(
        data=real_data,
        experiment=exp_real_3,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=False,  # no simulator oracle for real data
    )
    artifacts_by_name[exp_real_3.name] = art_real
    
    # synthetic NHANES
    synth_preproc_3 = NHANESPreprocessor(
        exclude_downstream=False,
        impute_strategy="median",
    )
    synth_model_3 = Cox(alpha=1)  # alpha to avioid overflow

    exp_synth_3 = Experiment(
        name="cox_synthetic_nhanes_naive",
        model=synth_model_3,
        preprocessor=synth_preproc_3,
        T_eval=T_eval_months,
        metadata={"dataset": "nhanes_synthetic"},
    )

    art_synth = evaluate_experiment(
        data=synth_data,
        experiment=exp_synth_3,
        sensitive_attr=sensitive_attr,
        sensitive_levels=None,
        verbose=verbose,
        test_causal=True
    )
    artifacts_by_name[exp_synth_3.name] = art_synth

    return artifacts_by_name
