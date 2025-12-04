from .plots import generate_plots
from .report import generate_report
from pathlib import Path
from ..common import get_experiment_artifacts_path_root


def main():
    # find all experiments in results directory and generate plots for each
    experiment_paths = [folder for folder in get_experiment_artifacts_path_root().iterdir()]

    for experiment_path in experiment_paths:
        print(f"Generating plots for experiment: {experiment_path.name}")
        generate_plots(experiment_path.name)

    print("generating report...")
    generate_report(experiment_paths)

    

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("config.env")
    main()

    