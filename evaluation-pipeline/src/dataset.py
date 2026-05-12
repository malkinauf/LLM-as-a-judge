import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

REQUIRED_DATASET_FIELDS = {"id", "question", "answer", "y_true"}


def save_dataset_to_file(
    dataset: list[dict[str, Any]],
    path: str,
    overwrite: bool = False,
) -> None:
    file_path = Path(path)

    if file_path.exists() and not overwrite:
        logger.warning("File already exists: %s", file_path)
        logger.info("Skipping save to avoid overwrite.")
        return
    if not isinstance(dataset, list):
        raise ValueError("Invalid dataset format. Expected a list of dictionaries.")
    if not all(isinstance(item, dict) for item in dataset):
        raise ValueError("Invalid dataset format. Expected a list of dictionaries.")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


def load_dataset_from_file(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError("Invalid dataset format. Expected a list.")

    for idx, row in enumerate(dataset):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid dataset row at index {idx}: expected an object.")
        missing_fields = REQUIRED_DATASET_FIELDS.difference(row.keys())
        if missing_fields:
            raise ValueError(
                f"Invalid dataset row at index {idx}: missing fields {sorted(missing_fields)}"
            )

    return dataset
