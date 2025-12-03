from matplotlib import pyplot as plt
from ..common import (
    get_experiment_artifacts_path, 
    get_analysis_path, 
    ExperimentArtifacts
)

def generate_index_plot(experiment_name: str):
    # we want to see how well the index performs vs the gt
    # so we will make a scatter plot of predicted vs gt and a line y=x
    # also reporting the metrics for reference in the legend
    
    # load artifacts
    artifacts = ExperimentArtifacts.load(
        get_experiment_artifacts_path(experiment_name)
    )
    
    # check if its a synthetic experiment
    if 'gt_risk_index' not in artifacts.data:
        print(f"Experiment {experiment_name} does not have ground truth risk index data. Skipping index plot.")
        return
    
    # plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        artifacts.data['gt_risk_index'],
        artifacts.data['predicted_risk_index'],
        alpha=0.5,
        label=f"Index Plot for {experiment_name}"
    )
    
    min_val = min(
        min(artifacts.data['gt_risk_index']),
        min(artifacts.data['predicted_risk_index'])
    )
    max_val = max(
        max(artifacts.data['gt_risk_index']),
        max(artifacts.data['predicted_risk_index'])
    )
    
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
    plt.xlabel('Ground Truth Index')
    plt.ylabel('Predicted Index')
    plt.title('Predicted vs Ground Truth Index')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_analysis_path(experiment_name) / 'index_plot.png')
    
    
    
def generate_plots(experiment_name: str):
    get_analysis_path(experiment_name).mkdir(parents=True, exist_ok=True)
    
    generate_index_plot(experiment_name)