from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).parent / "configs"


REQUIRED_CONFIG_KEYS = {
    "name",
    "output_dataset_name",
    "source",
    "samples",
}


def validate_dataset_config(config: dict[str, Any]) -> None:
    """
    Validate the structure of a dataset configuration.
    """

    missing_keys = REQUIRED_CONFIG_KEYS - set(config.keys())

    if missing_keys:
        raise ValueError(
            f"Dataset config is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    # Validate source
    source = config["source"]

    if not isinstance(source, dict):
        raise ValueError("'source' must be a mapping.")

    if "type" not in source:
        raise ValueError(
            "Dataset config source is missing 'type'."
        )

    # Validate samples
    samples = config["samples"]

    if not isinstance(samples, list) or not samples:
        raise ValueError(
            "'samples' must be a non-empty list."
        )

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Sample rule {i} must be a mapping."
            )

        for field in ("question", "model_response", "y_true"):
            if field not in sample:
                raise ValueError(
                    f"Sample rule {i} is missing '{field}'."
                )

            field_config = sample[field]

            if not isinstance(field_config, dict):
                raise ValueError(
                    f"'{field}' in sample rule {i} "
                    "must be a mapping."
                )

            if (
                "source" not in field_config
                and "value" not in field_config
            ):
                raise ValueError(
                    f"'{field}' in sample rule {i} must define "
                    "'source' or 'value'."
                )


def load_dataset_config(name: str) -> dict[str, Any]:
    """
    Load and validate a dataset configuration.
    """

    config_path = CONFIG_DIR / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Dataset config must be a YAML object: {config_path}"
        )

    validate_dataset_config(config)

    return config
