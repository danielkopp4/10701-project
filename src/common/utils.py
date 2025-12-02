from pathlib import Path
import os


def get_experiment_artifacts_path(experiment_name: str):
    return Path(os.getenv('EXPERIMENT_ARTIFACTS_PATH', 'experiment_results')) / experiment_name

def get_analysis_path(experiment_name: str):
    return Path(os.getenv('ANALYSIS_PATH', 'analysis_results')) / experiment_name