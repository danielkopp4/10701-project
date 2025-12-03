from .plots import generate_plots
import os
from pathlib import Path


def main():
    # find all experiments in results directory and generate plots for each
    results_dir = Path(os.getenv('EXPERIMENT_ARTIFACTS_PATH', 'experiment_results'))
    
    for experiment_dir in results_dir.iterdir():
        print(f"Generating plots for experiment: {experiment_dir.name}")
        generate_plots(experiment_dir.name)
    

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv("config.env")
    main()

    