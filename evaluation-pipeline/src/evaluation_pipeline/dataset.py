import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


REQUIRED_DATASET_KEYS = {"id", "question", "model_response", "y_true"}

VALID_LABELS = {
    "truthful",
    "not_truthful"
}


def validate_dataset_entry(item: dict[str, Any], index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"Dataset item {index} must be a dictionary.")

    missing_keys = REQUIRED_DATASET_KEYS - item.keys()
    if missing_keys:
        raise ValueError(
            f"Dataset item {index} is missing required keys: {missing_keys}"
        )

    if item["y_true"] not in VALID_LABELS:
        raise ValueError(
            f"Dataset item {index} has invalid label: {item['y_true']}"
        )

    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"Dataset item {index} has empty question.")

    if not isinstance(item["model_response"], str) or not item["model_response"].strip():
        raise ValueError(f"Dataset item {index} has empty model_response.")


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of dictionaries.")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    for i, item in enumerate(dataset):
        validate_dataset_entry(item, i)


def save_dataset_to_file(
    dataset: list[dict[str, Any]],
    path: str,
    overwrite: bool = False,
) -> None:
    validate_dataset(dataset)

    file_path = Path(path)

    if file_path.exists() and not overwrite:
        logger.warning(f"File already exists: {file_path}")
        logger.info("Skipping save to avoid overwrite.")
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved to {file_path}")


def load_dataset_from_file(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in dataset file: {file_path}"
        ) from e

    validate_dataset(dataset)

    logger.info(f"Dataset loaded from {file_path}")

    return dataset
