from typing import Any


def map_fields(
    row: dict[str, Any],
    mapping: dict[str, str],
) -> dict[str, Any]:
    """
    Map source dataset fields to the canonical dataset schema.

    Args:
        row:
            Original dataset row.
        mapping:
            Mapping from canonical field names to source field names.

            Example:
            {
                "question": "prompt",
                "model_response": "response",
            }

    Returns:
        Dictionary containing mapped canonical fields.
    """

    mapped = {}

    for target_field, source_field in mapping.items():
        if source_field not in row:
            raise ValueError(
                f"Source field '{source_field}' "
                f"not found in dataset row."
            )

        mapped[target_field] = row[source_field]

    return mapped


def map_label(
    row: dict[str, Any],
    label_config: dict[str, Any],
) -> str:
    """
    Map a source dataset label to the canonical y_true label.

    Args:
        row:
            Original dataset row.
        label_config:
            Label mapping configuration.

            Example:
            {
                "source": "is_safe",
                "values": {
                    True: "safe",
                    False: "not_safe",
                },
            }

    Returns:
        Canonical label.
    """

    source_field = label_config["source"]

    if source_field not in row:
        raise ValueError(
            f"Label source field '{source_field}' "
            f"not found in dataset row."
        )

    raw_label = row[source_field]
    label_values = label_config["values"]

    if raw_label not in label_values:
        raise ValueError(
            f"Unknown label value '{raw_label}' "
            f"for source field '{source_field}'."
        )

    return label_values[raw_label]


def map_entry(
    row: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert one source dataset row into canonical fields.

    The returned entry does not yet contain generated fields
    such as id or dataset name.
    """

    mapped = map_fields(
        row=row,
        mapping=config["mapping"],
    )

    mapped["y_true"] = map_label(
        row=row,
        label_config=config["label"],
    )

    return mapped