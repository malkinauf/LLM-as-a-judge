from typing import Any, Dict, Optional


def log_experiment(
    *,
    project_name: str,
    run_name: str,
    config: Dict[str, Any],
    df_results,
    df_summary_metrics,
    df_second_level_metrics=None,
    df_second_level_cases=None,
    confusion_figure=None,
    enabled: bool = True,
) -> None:
    """Log experiment results to Weights & Biases.

    Keeping W&B here keeps the notebook focused on experiment orchestration.
    """
    if not enabled:
        print("W&B logging disabled.")
        return

    import wandb

    wandb.login()
    run = wandb.init(project=project_name, name=run_name, config=config)

    try:
        wandb.log({
            "raw_results": wandb.Table(dataframe=df_results),
            "summary_metrics": wandb.Table(dataframe=df_summary_metrics),
        })

        if df_second_level_metrics is not None:
            wandb.log({
                "second_level_metrics": wandb.Table(dataframe=df_second_level_metrics),
            })

        if df_second_level_cases is not None:
            wandb.log({
                "second_level_cases": wandb.Table(dataframe=df_second_level_cases),
            })

        if not df_summary_metrics.empty:
            scalar_names = [
                "accuracy", "precision", "recall", "f1",
                "cohen_kappa", "mcc", "coverage",
            ]
            row = df_summary_metrics.iloc[0]
            scalars = {name: float(row[name]) for name in scalar_names if name in row}
            if scalars:
                wandb.log(scalars)

        if confusion_figure is not None:
            wandb.log({"confusion_matrix": wandb.Image(confusion_figure)})
    finally:
        run.finish()