from typing import Dict

from ..common import Experiment, ExperimentArtifacts, get_experiment_artifacts_path
from .models import Cox, GradientBoosting, RandomSurvivalForest, SVM
from .preprocessors import NHANESPreprocessor
from .evaluate import evaluate_experiment
from ..get_data.dataset_api import load_nhanes_survival  # your real-data loader
from .synthetic_data.simulator import (
    load_nhanes_survival_simulated
)
from copy import deepcopy

def run_experiments(
    nhanes_csv_path: str,
    synthetic_n: int = 5000,
    T_eval_years: float = 10.0,
    A_strata: int = 100,
    verbose: bool = True,
) -> Dict[str, ExperimentArtifacts]:
    T_eval_months = T_eval_years * 12.0
    artifacts_by_name: Dict[str, ExperimentArtifacts] = {}

    real_data = load_nhanes_survival(csv_path=nhanes_csv_path)
    synth_data = load_nhanes_survival_simulated(
        n=synthetic_n,
        real_csv=nhanes_csv_path,
        use_real_baseline=True
    )
    
    # make sure experiment artifacts folder exists
    get_experiment_artifacts_path("placeholder").parent.mkdir(parents=True, exist_ok=True)
    
    def evaluate_experiments(datasets, preprocessors, models, A_strata=50, T_eval=T_eval_months, target_feature = "BMXWT", name_prefix=""):
        for dataset_name, dataset in datasets:
            for preproc_name, preprocessor in preprocessors:
                for model_name, model in models:
                    model_instance = deepcopy(model) # make sure to not share state
                    preprocessor_instance = deepcopy(preprocessor)
                    exp_name = f"{name_prefix}{model_name}_{dataset_name}_{preproc_name}"
                    experiment = Experiment(
                        name=exp_name,
                        model=model_instance,
                        preprocessor=preprocessor_instance,
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
        ("nhanes-synthetic", synth_data),
        ("nhanes-real", real_data),
    ]
    
    preprocessors = [
        ("Exclude Downstream", NHANESPreprocessor(exclude_downstream=True, exclude_BMI=True, exclude_waist=True)),
        ("Intake Only", NHANESPreprocessor(include_only_intake=True, exclude_downstream=True, exclude_BMI=True)),
        ("Intake w.o. W-History", NHANESPreprocessor(include_only_intake=True, exclude_downstream=True, exclude_BMI=True, exclude_weight_history=True)),
        ("Intake + Basic Exam", NHANESPreprocessor(include_only_intake_and_basic=True, exclude_downstream=True, exclude_BMI=True, exclude_waist=True)),
    ]
    
    models = [
        ("random-forest", RandomSurvivalForest()),
        ("cox", Cox(alpha=0.2)),
        ("svm", SVM(alpha=1.0, rank_ratio=1.0)),
        # ("gradient-boost", GradientBoosting()),
    ]
    
    evaluate_experiments(
        datasets, 
        preprocessors, 
        models, 
        A_strata=A_strata
    )
    
    # BMXBMI as target instead of weight
    preprocessors = [
        ("BMI Only", NHANESPreprocessor(include_only_bmi=True)), # standard approach
    ]
    
    evaluate_experiments(
        datasets, 
        preprocessors, 
        models,
        A_strata=A_strata,
        target_feature="BMXBMI",
        name_prefix="bmi-as-target_"
    )
    
    # BMXWAIST as target instead of weight
    preprocessors = [
        ("WtH Only", NHANESPreprocessor(include_only_waist_to_height=True)),
        ("Intake + Basic Exam (waist circ.)", NHANESPreprocessor(include_only_intake_and_basic=True, exclude_downstream=True, exclude_weight=True, exclude_BMI=True)), # uses waist instead of weight
        ("Intake + Basic Exam w.o. W-History (waist circ.)", NHANESPreprocessor(include_only_intake_and_basic=True, exclude_downstream=True, exclude_weight=True, exclude_weight_history=True, exclude_BMI=True)), # uses waist instead of weight
    ]
    
    evaluate_experiments(
        datasets, 
        preprocessors, 
        models,
        A_strata=A_strata,
        target_feature="BMXWAIST",
        name_prefix="waist-as-target_"
    )
    
    evaluate_experiments(
        datasets, 
        [("include-all", NHANESPreprocessor())], 
        [
            ("cox", Cox(alpha=0.2)), 
            # ("cox-strong-reg", Cox(alpha=1.0)), 
            # ("gradient-boost", GradientBoosting()),
            ("random-forest", RandomSurvivalForest()),
            ("svm", SVM(alpha=1.0, rank_ratio=1.0)),
        ], 
        A_strata=A_strata, name_prefix="naive_"
    )
    
    

    return artifacts_by_name
