from typing import Dict

from ..common import Experiment, ExperimentArtifacts, get_experiment_artifacts_path
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
    target_feature: str = "BMXWT",
    verbose: bool = True,
) -> Dict[str, ExperimentArtifacts]:
    T_eval_months = T_eval_years * 12.0
    artifacts_by_name: Dict[str, ExperimentArtifacts] = {}

    real_data = load_nhanes_survival(csv_path=nhanes_csv_path)
    synth_data = load_nhanes_survival_simulated(
        n=synthetic_n
    )
    
    # make sure experiment artifacts folder exists
    get_experiment_artifacts_path("placeholder").parent.mkdir(parents=True, exist_ok=True)
    
    def evaluate_experiments(datasets, preprocessors, models, A_strata=50, T_eval=T_eval_months, name_prefix=""):
        for dataset_name, dataset in datasets:
            for preproc_name, preprocessor in preprocessors:
                for model_name, model in models:
                    exp_name = f"{name_prefix}{model_name}_{dataset_name}_{preproc_name}"
                    experiment = Experiment(
                        name=exp_name,
                        model=model,
                        preprocessor=preprocessor,
                        T_eval=T_eval,
                        A_strata=A_strata,
                        metadata={"dataset": dataset_name},
                    )
                    
                    art = evaluate_experiment(
                        data=dataset,
                        experiment=experiment,
                        target_feature=target_feature,
                        verbose=verbose,
                        test_causal=(dataset_name == "nhanes-synthetic")
                    )
                    art.save(
                        get_experiment_artifacts_path(experiment.name)
                    )
                    artifacts_by_name[experiment.name] = art
    
    datasets = [
        ("nhanes-real", real_data),
        ("nhanes-synthetic", synth_data),
    ]
    
    preprocessors = [
        ("exclude-downstream", NHANESPreprocessor(exclude_downstream=True)),
        ("exclude-all", NHANESPreprocessor(exclude_all=True)), # standard approach
        # ("include_all", False), # requires higher alpha 
    ]
    
    models = [
        ("cox", Cox(alpha=0.1)),
    ]
    
    evaluate_experiments(
        datasets, 
        preprocessors, 
        models, 
        A_strata=50
    )
    evaluate_experiments(
        datasets, 
        [("include-all", NHANESPreprocessor(exclude_downstream=False))], 
        [("cox-strong-reg", Cox(alpha=1.0))], 
        A_strata=50, name_prefix="naive_"
    )
    
    

    return artifacts_by_name
