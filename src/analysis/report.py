from typing import List
import pandas as pd
from pathlib import Path
from ..common import (
    ExperimentArtifacts, 
    get_overall_analysis_path
)

COLUMN_LABELS = {
    "model": "Model",
    "dataset": "Dataset",
    "name": "Experiment",
    "test_cindex": "Test C-index",
    "train_cindex": "Train C-index",
    "pred_vs_gt_index_corr": "Index Correlation",
    "pred_vs_gt_index_rmse": "Index RMSE",
}

def human_label(col: str) -> str:
    if col in COLUMN_LABELS:
        return COLUMN_LABELS[col]
    return col.replace("_", " ")


def create_table(table_name: str, experiment_artifact_paths: List[Path], sort_by: str, metrics_to_include: List[str], include_name: bool = True, asc: bool = True, separate_name: bool = True, **formatting_kwargs):
    assert sort_by in metrics_to_include

    # load all results to a table
    cols = {metric: [] for metric in metrics_to_include}
    
    if include_name:
        if separate_name:
            cols["model"] = []
            cols["dataset"] = []
            cols["name"] = []
        else:
            cols["name"] = []

    for experiment_artifact_path in experiment_artifact_paths:
        experiment_artifact = ExperimentArtifacts.load(
            experiment_artifact_path
        )

        if include_name:
            name = experiment_artifact.experiment.name
            if separate_name:
                cols["model"].append(name.split("_")[-3])
                cols["dataset"].append(name.split("_")[-2])
                cols["name"].append(name.split("_")[-1])
            else:
                cols["name"].append(name)

        for metric in metrics_to_include:
            if metric in experiment_artifact.metrics:
                cols[metric].append(experiment_artifact.metrics[metric])
            else:
                cols[metric].append(None)
                print('experiment', experiment_artifact_path, " dindt have", metric)

    columns = []

    if include_name:
        if separate_name:
            columns += ["model", "dataset", "name"]
        else:
            columns += ["name"]

    columns += metrics_to_include

    df = pd.DataFrame(cols, columns=columns)

    df = df.sort_values(by=sort_by, ascending=asc)

    table_root = get_overall_analysis_path()
    table_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_root / f"{table_name}.csv", index=False)
    df = df.rename(columns={c: human_label(c) for c in df.columns})
    
    original_formatters = formatting_kwargs.pop("formatters", None)
    if original_formatters is not None:
        latex_formatters = {}
        for col, func in original_formatters.items():
            pretty_col = human_label(col)
            latex_formatters[pretty_col] = func
        formatting_kwargs["formatters"] = latex_formatters
    
    df.to_latex(table_root / f"{table_name}.tex", index=False, **formatting_kwargs)


def generate_report(experiment_paths: List[Path]):
    # export tables comparing the models 
    create_table(
        "sort_by_test_cindex", 
        experiment_paths, 
        sort_by="test_cindex",
        metrics_to_include=[
            "train_cindex", 
            "test_cindex", 
            "pred_vs_gt_index_corr", 
            "pred_vs_gt_index_rmse"
        ],
        asc=False,
        float_format="{:.3f}".format,
        formatters={
            "dataset": lambda x: "Synthetic" if x.find("synthetic") else "Real",
            "model": lambda x: {"random-forest": "RF", "gradient-boost": "GB", "cox": "Cox"}.get(x)
        }
    )

    create_table(
        "sort_by_pred_vs_gt_index_corr", 
        experiment_paths, 
        sort_by="pred_vs_gt_index_corr",
        metrics_to_include=[
            "train_cindex", 
            "test_cindex", 
            "pred_vs_gt_index_corr", 
            "pred_vs_gt_index_rmse"
        ],
        asc=False,
        float_format="{:.3f}".format,
        formatters={
            "dataset": lambda x: "Synthetic" if x.find("synthetic") else "Real",
            "model": lambda x: {"random-forest": "RF", "gradient-boost": "GB", "cox": "Cox"}.get(x)
        }
    )
