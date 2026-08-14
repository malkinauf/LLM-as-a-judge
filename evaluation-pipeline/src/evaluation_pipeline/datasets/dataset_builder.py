import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset

from evaluation_pipeline.datasets.config_loader import (
    load_dataset_config,
)
from evaluation_pipeline.datasets.mapper import map_entry
from evaluation_pipeline.datasets.schema import validate_dataset


def load_source(
    source_config: dict[str, Any],
):
    """
    Load a raw dataset from the configured source.

    Currently supports Hugging Face datasets and JSON files.

    Args:
        source_config: Source configuration containing the
            source type and source-specific parameters.

    Returns:
        The loaded raw dataset.

    Raises:
        FileNotFoundError: If a local JSON source does not exist
            and no download URL is provided.
        ValueError: If the configured source type or JSON structure
            is not supported.
    """
    source_type = source_config["type"]

    if source_type == "huggingface":
        dataset_name = source_config["dataset"]
        subset = source_config.get("subset")
        split = source_config["split"]

        return load_dataset(
            dataset_name,
            subset,
            split=split,
        )

    if source_type == "json":
        path = Path(source_config["path"])
        url = source_config.get("url")

        if not path.exists():
            if url is None:
                raise FileNotFoundError(
                    f"Dataset source not found: {path}"
                )

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            response = requests.get(
                url,
                timeout=60,
            )
            response.raise_for_status()

            path.write_bytes(
                response.content
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "JSON dataset must contain a mapping "
                "of groups to dataset entries."
            )

        rows = []

        for entries in data.values():
            if not isinstance(entries, list):
                raise ValueError(
                    "Each JSON dataset group must contain "
                    "a list of entries."
                )

            rows.extend(entries)

        return rows

    raise ValueError(
        f"Unsupported dataset source type: '{source_type}'"
    )


def _has_agreement(
    row: dict[str, Any],
    fields: list[str],
) -> bool:
    """
    Check whether all configured fields contain the same value.

    Args:
        row: Raw dataset row.
        fields: Fields whose values must agree.

    Returns:
        True if all configured values are identical,
        otherwise False.

    Raises:
        ValueError: If no agreement fields are configured.
        KeyError: If a configured field is missing from the row.
    """
    if not fields:
        raise ValueError(
            "Agreement fields must not be empty."
        )

    values = [
        row[field]
        for field in fields
    ]

    return len(set(values)) == 1


def _balanced_sample(
    dataset: list[dict[str, Any]],
    n_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Create a balanced random sample from a binary dataset.

    Samples are selected randomly across the complete candidate
    dataset, with an equal number of entries from each label.

    Args:
        dataset: Canonical dataset samples.
        n_samples: Total number of samples to select.
        seed: Random seed for reproducible sampling.

    Returns:
        A shuffled dataset containing an equal number of samples
        from both labels.

    Raises:
        ValueError: If n_samples is invalid, the dataset does not
            contain exactly two labels, or one label contains too
            few samples.
    """
    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    if n_samples % 2 != 0:
        raise ValueError(
            "Balanced binary sampling requires "
            "an even n_samples."
        )

    groups = defaultdict(list)

    for sample in dataset:
        groups[sample["y_true"]].append(
            sample
        )

    if len(groups) != 2:
        raise ValueError(
            "Balanced binary sampling requires "
            "exactly two labels."
        )

    samples_per_label = n_samples // 2
    rng = random.Random(seed)

    selected = []

    for label, samples in groups.items():
        if len(samples) < samples_per_label:
            raise ValueError(
                f"Not enough samples for label '{label}'. "
                f"Required {samples_per_label}, "
                f"available {len(samples)}."
            )

        selected.extend(
            rng.sample(
                samples,
                samples_per_label,
            )
        )

    rng.shuffle(selected)

    return selected


def build_dataset(
    name: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Build a dataset in the canonical evaluation format.

    The function loads the configured raw dataset, optionally
    filters rows by agreement, maps raw rows to canonical samples,
    applies the configured sampling strategy, assigns metadata,
    and validates the final dataset.

    Args:
        name: Name of the dataset configuration.
        n_samples: Number of output samples.
        seed: Random seed for reproducible sampling.

    Returns:
        A validated list of dataset samples in the canonical
        evaluation format.

    Raises:
        ValueError: If n_samples is invalid, not enough candidate
            samples are available, or the configured sampling
            strategy cannot be applied.
    """
    if n_samples <= 0:
        raise ValueError(
            "n_samples must be positive."
        )

    config = load_dataset_config(
        name
    )

    raw_dataset = load_source(
        config["source"]
    )

    candidates = []

    agreement = config.get(
        "agreement"
    )

    for row in raw_dataset:
        if agreement is not None:
            if not _has_agreement(
                row=row,
                fields=agreement["fields"],
            ):
                continue

        mapped_samples = map_entry(
            row=row,
            config=config,
        )

        candidates.extend(
            mapped_samples
        )

    if n_samples > len(candidates):
        raise ValueError(
            f"Requested {n_samples} samples, but only "
            f"{len(candidates)} samples are available."
        )

    sampling = config.get(
        "sampling",
        {},
    )

    strategy = sampling.get(
        "strategy",
        "random",
    )

    if strategy == "balanced":
        result = _balanced_sample(
            dataset=candidates,
            n_samples=n_samples,
            seed=seed,
        )

    elif strategy == "random":
        rng = random.Random(seed)

        result = rng.sample(
            candidates,
            n_samples,
        )

    else:
        raise ValueError(
            f"Unsupported sampling strategy: "
            f"'{strategy}'"
        )

    result = [
        {
            "id": f"{config['name']}_{i}",
            "dataset": config[
                "output_dataset_name"
            ],
            **sample,
        }
        for i, sample in enumerate(
            result
        )
    ]

    validate_dataset(
        result
    )

    return result
