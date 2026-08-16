from typing import Any


REQUIRED_FIELDS = {
    "id",
    "dataset",
    "question",
    "model_response",
    "y_true",
}


def validate_dataset_entry(
    item: dict[str, Any],
    index: int,
) -> None:
    """
    Validate one canonical dataset sample.
    """
    missing_fields = REQUIRED_FIELDS - set(item.keys())

    if missing_fields:
        raise ValueError(
            f"Dataset item {index} is missing fields: "
            f"{sorted(missing_fields)}"
        )

    for field in (
        "id",
        "dataset",
        "question",
        "model_response",
        "y_true",
    ):
        value = item[field]

        if not isinstance(value, str):
            raise ValueError(
                f"Dataset item {index} field '{field}' "
                "must be a string."
            )

        if not value.strip():
            raise ValueError(
                f"Dataset item {index} field '{field}' "
                "must not be empty."
            )


def validate_dataset(
    dataset: list[dict[str, Any]],
) -> None:
    """
    Validate a canonical evaluation dataset.
    """
    if not dataset:
        raise ValueError(
            "Dataset is empty."
        )

    for index, item in enumerate(dataset):
        validate_dataset_entry(
            item,
            index,
        )