from typing import Any


def _resolve_value(
    row: dict[str, Any],
    field_config: dict[str, Any],
) -> Any:
    """
    Resolve a value from a sample field configuration.

    Supported forms:
    - source: read value from the raw dataset row
    - value: use a static value
    - take: first -> take the first element from a list
    """

    if "source" in field_config:
        source_field = field_config["source"]

        if source_field not in row:
            raise ValueError(
                f"Source field '{source_field}' "
                "not found in dataset row."
            )

        value = row[source_field]

    elif "value" in field_config:
        value = field_config["value"]

    else:
        raise ValueError(
            "Field config must define 'source' or 'value'."
        )

    if field_config.get("take") == "first":
        if not value:
            raise ValueError(
                "Cannot take first element from an empty value."
            )

        value = value[0]

    if "values" in field_config:
        value_mapping = field_config["values"]

        if value not in value_mapping:
            raise ValueError(
                f"Value '{value}' not found in value mapping."
            )

        value = value_mapping[value]

    return value


def map_entry(
    row: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Convert one raw dataset row into one or more
    canonical dataset samples.

    The number of returned samples is determined by
    the number of rules defined in config['samples'].
    """

    mapped_samples = []

    for sample_config in config["samples"]:
        mapped_sample = {
            "question": _resolve_value(
                row=row,
                field_config=sample_config["question"],
            ),
            "model_response": _resolve_value(
                row=row,
                field_config=sample_config["model_response"],
            ),
            "y_true": _resolve_value(
                row=row,
                field_config=sample_config["y_true"],
            ),
        }

        mapped_samples.append(mapped_sample)

    return mapped_samples
