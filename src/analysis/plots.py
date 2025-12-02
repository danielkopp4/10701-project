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
    artifacts = ExperimentArtifacts.load_from_path(
        get_experiment_artifacts_path(experiment_name)
    )
    
    # plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        artifacts.data['artifact_index_gt'],
        artifacts.data['artifact_index_pred'],
        alpha=0.5,
        label=f"Index Plot for {experiment_name}"
    )
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x')
    plt.xlabel('Ground Truth Index')
    plt.ylabel('Predicted Index')
    plt.title('Predicted vs Ground Truth Index')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(get_analysis_path(experiment_name) / 'index_plot.png')
    
    # return plot
    
    
def generate_plots(experiment_name: str):
    get_analysis_path(experiment_name).mkdir(parents=True, exist_ok=True)