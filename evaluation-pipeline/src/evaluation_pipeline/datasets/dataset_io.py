import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from evaluation_pipeline.datasets.schema import validate_dataset


logger = logging.getLogger(__name__)

def _get_project_root() -> Path:
    """
    Find the project root containing pyproject.toml.
    """
    current = Path(__file__).resolve().parent

    for path in (current, *current.parents):
        if (path / "pyproject.toml").is_file():
            return path

    raise FileNotFoundError(
        "Could not find project root."
    )

def get_prepared_dataset_path(
    dataset_id: str,
    n_samples: int 
) -> Path:
    return (
        _get_project_root()
        / "datasets"
        / "prepared"
        / f"{dataset_id}_{n_samples}.json"
    )

def save_dataset_to_file(
    dataset: list[dict[str, Any]],
    path: str | Path,
    overwrite: bool = False,
    raise_on_exists: bool = False,
) -> None:
    """
    Save a validated dataset to a JSON file.
    """

    validate_dataset(dataset)
    file_path = Path(path)

    if file_path.exists() and not overwrite:
        message = f"File already exists: {file_path}"

        if raise_on_exists:
            raise FileExistsError(message)

        logger.warning(message)
        logger.info(
            "Skipping save to avoid overwriting existing file."
        )
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(
            dataset,
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"Dataset saved to {file_path}")

def load_prepared_dataset(
    dataset_id: str,
) -> list[dict[str, Any]]:
    """
    Load a prepared dataset by its dataset ID.
    """
    if Path(dataset_id).name != dataset_id:
        raise ValueError(
            "dataset_id must be a dataset name, not a path."
        )

    file_path = (
        _get_project_root()
        / "datasets"
        / "prepared"
        / f"{dataset_id}.json"
    )

    return load_dataset_from_file(
        file_path
    )

def load_dataset_from_file(
    path: str | Path,
) -> list[dict[str, Any]]:
    """
    Load and validate a prepared dataset from a JSON file.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in dataset file: {file_path}"
        ) from e

    validate_dataset(dataset)

    logger.info(
        f"Dataset loaded from {file_path}"
    )

    return dataset


def preview_dataset(
    dataset: list[dict[str, Any]],
) -> None:
    """
    Print a short summary and one example from a dataset.
    """

    validate_dataset(dataset)

    labels = [
        item["y_true"]
        for item in dataset
    ]

    print("Total samples:", len(dataset))
    print("Labels:", Counter(labels))

    print("\nExample:\n")

    print(
        json.dumps(
            dataset[0],
            indent=2,
            ensure_ascii=False,
        )
    )
