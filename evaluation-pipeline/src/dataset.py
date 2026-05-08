import json
import logging
from pathlib import Path
from typing import Any

# ToDo Check fromat dataset, should be list of dicts with keys "question", "answer", "label"

logger = logging.getLogger(__name__)

def save_dataset_to_file(
        dataset: list[dict[str, Any]],
        path: str,
        overwrite: bool = False
        ) -> None:
    file_path = Path(path)

    if file_path.exists() and not overwrite:
        logger.warning(f"File already exists: {file_path}")
        logger.info("Skipping save to avoid overwrite.")
        return
    if not isinstance(dataset, list):
        logger.error("Dataset must be a list of dictionaries.")
        raise ValueError("Invalid dataset format. Expected a list of dictionaries.")
    if not all(isinstance(item, dict) for item in dataset):
        logger.error("All items in the dataset must be dictionaries.")
        raise ValueError("Invalid dataset format. Expected a list of dictionaries.")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved to {file_path}")


def load_dataset_from_file(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)

    if not file_path.exists():
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Dataset loaded from {file_path}")

    return dataset