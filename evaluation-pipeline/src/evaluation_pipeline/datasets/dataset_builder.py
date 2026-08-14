from typing import Any

from datasets import load_dataset

from evaluation_pipeline.datasets.config_loader import (
    load_dataset_config,
)


def load_source(
    source_config: dict[str, Any],
):
    source_type = source_config["type"]

    if source_type == "huggingface":
        dataset_name = source_config["dataset"]
        split = source_config["split"]

        return load_dataset(dataset_name)[split]

    raise ValueError(
        f"Unsupported dataset source type: '{source_type}'"
    )


def build_dataset(
    name: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    config = load_dataset_config(name)
    raw_dataset = load_source(config["source"])
    first_row = raw_dataset[0]
    mapped_row = _map_entry(
        row=first_row,
        config=config,
    )
    return mapped_row

def _map_entry(
    row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    mapping = config["mapping"]

    return {
        "question": row[mapping["question"]],
        "model_response": row[mapping["model_response"]],
    }
