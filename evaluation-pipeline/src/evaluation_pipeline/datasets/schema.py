from typing import Any

REQUIRED_DATASET_KEYS = {
    "id",
    "dataset",
    "question",
    "model_response",
    "y_true"
}

VALID_LABELS = {
    "truthful",
    "not_truthful",
    "safe",
    "not_safe",
}


def validate_dataset_entry(item: dict[str, Any], index: int) -> None:
    """
    Validate a single dataset entry.

    Args:
        item: Dataset entry to validate.
        index: Position of the entry in the dataset.

    Raises:
        ValueError: If the entry has an invalid format,
            missing required keys, or invalid values.
    """

    if not isinstance(item, dict):
        raise ValueError(f"Dataset item {index} must be a dictionary.")

    missing_keys = REQUIRED_DATASET_KEYS - set(item.keys())
    if missing_keys:
        raise ValueError(
            f"Dataset item {index} is missing required keys: {sorted(missing_keys)}"
        )
    if not isinstance(item["id"], str) or not item["id"].strip():
        raise ValueError(f"Dataset item {index} has invalid id: {item['id']}")

    if not isinstance(item["dataset"], str) or not item["dataset"].strip():
        raise ValueError(
            f"Dataset item {index} has invalid dataset: {item['dataset']}")

    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"Dataset item {index} has empty question.")

    if not isinstance(item["model_response"], str) or not item["model_response"].strip():
        raise ValueError(f"Dataset item {index} has empty model_response.")

    if not isinstance(item["y_true"], str) or item["y_true"] not in VALID_LABELS:
        raise ValueError(
            f"Dataset item {index} has invalid label: {item['y_true']}"
        )


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    """
    Validate the entire dataset.

    Checks that the dataset is non-empty and that every
    entry conforms to the expected schema.

    Args:
        dataset: Dataset to validate.

    Raises:
        ValueError: If the dataset is empty or contains
            invalid entries.
    """

    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list.")

    if not dataset:
        raise ValueError("Dataset is empty.")

    for i, item in enumerate(dataset):
        validate_dataset_entry(item, i)
