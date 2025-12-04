from matplotlib import pyplot as plt
import matplot2tikz

from ..common import (
    get_experiment_artifacts_path,
    get_analysis_path,
    ExperimentArtifacts,
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

    analysis_path = get_analysis_path(experiment_name)
    analysis_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(
        artifacts.data["gt_risk_index"],
        artifacts.data["predicted_risk_index"],
        alpha=0.5,
        label=f"Index Plot for {experiment_name}",
    )

    min_val = min(
        min(artifacts.data["gt_risk_index"]),
        min(artifacts.data["predicted_risk_index"]),
    )
    max_val = max(
        max(artifacts.data["gt_risk_index"]),
        max(artifacts.data["predicted_risk_index"]),
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        label="y=x",
    )
    ax.set_xlabel("Ground Truth Index")
    ax.set_ylabel("Predicted Index")
    ax.set_title("Predicted vs Ground Truth Index")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    # file paths
    png_path = analysis_path / "index_plot.png"
    tikz_path = analysis_path / "index_plot.tex"

    # save PNG
    fig.savefig(png_path, dpi=300)

    # save TikZ/PGFPlots
    # matplot2tikz uses the current figure by default
    matplot2tikz.save(str(tikz_path))

    plt.close(fig)


def generate_plots(experiment_name: str):
    get_analysis_path(experiment_name).mkdir(parents=True, exist_ok=True)
    generate_index_plot(experiment_name)
