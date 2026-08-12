from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def log_experiment_to_wandb(
    *,
    project_name: str,
    config: dict[str, Any],
    df_results: pd.DataFrame,
    df_summary_metrics: pd.DataFrame,
    df_second_level_metrics: pd.DataFrame | None = None,
    df_second_level_cases: pd.DataFrame | None = None,
    confusion_matrix_fig: Any | None = None,
    results_dir: str | Path | None = None,
    login: bool = False,
    finish: bool = True,
):
    import wandb

    if login:
        wandb.login()

    run = wandb.init(
        project=project_name,
        config=config,
    )

    payload: dict[str, Any] = {
        "raw_results": wandb.Table(dataframe=df_results),
        "summary_metrics": wandb.Table(dataframe=df_summary_metrics),
    }

    if df_second_level_metrics is not None and not df_second_level_metrics.empty:
        payload["second_level_metrics"] = wandb.Table(
            dataframe=df_second_level_metrics,
        )

    if df_second_level_cases is not None and not df_second_level_cases.empty:
        payload["second_level_cases"] = wandb.Table(
            dataframe=df_second_level_cases,
        )

    wandb.log(payload)

    if confusion_matrix_fig is not None:
        wandb.log({"confusion_matrix": wandb.Image(confusion_matrix_fig)})

    if results_dir is not None:
        results_path = Path(results_dir)
        if results_path.exists():
            artifact = wandb.Artifact(
                name=f"{config.get('run_id', 'run')}_results",
                type="results",
                description="All experiment result files from the results folder",
            )
            artifact.add_dir(str(results_path))
            run.log_artifact(artifact)

            file_rows = []
            for file_path in sorted(results_path.rglob("*")):
                if file_path.is_file():
                    relative_path = file_path.relative_to(results_path)
                    file_rows.append(
                        {
                            "file": str(relative_path),
                            "size_bytes": file_path.stat().st_size,
                            "suffix": file_path.suffix,
                        }
                    )

            if file_rows:
                wandb.log({
                    "results_index": wandb.Table(
                        dataframe=pd.DataFrame(file_rows),
                    )
                })

    if finish:
        wandb.finish()

    return run
