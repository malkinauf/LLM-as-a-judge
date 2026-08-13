from typing import Any

from evaluation_pipeline.datasets.config_loader import (
    load_dataset_config,
)


def build_dataset(
    name: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    config = load_dataset_config(name)

    return []