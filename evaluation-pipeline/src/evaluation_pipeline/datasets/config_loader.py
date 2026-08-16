from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).parent / "configs"


REQUIRED_CONFIG_KEYS = {
    "name",
    "output_dataset_name",
    "source",
    "mapping",
}


SUPPORTED_MAPPING_TYPES = {
    "single",
    "multi",
    "aligned",
}


def validate_dataset_config(
    config: dict[str, Any],
) -> None:
    """
    Validate the basic dataset configuration structure.

    Args:
        config: Parsed dataset configuration.

    Raises:
        ValueError: If required fields are missing or invalid.
    """

    missing_keys = REQUIRED_CONFIG_KEYS - set(config.keys())

    if missing_keys:
        raise ValueError(
            f"Dataset config is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    source = config["source"]

    if not isinstance(source, dict):
        raise ValueError(
            "'source' must be a mapping."
        )

    if "type" not in source:
        raise ValueError(
            "Dataset config source is missing 'type'."
        )

    mapping = config["mapping"]

    if not isinstance(mapping, dict):
        raise ValueError(
            "'mapping' must be a mapping."
        )

    mapping_type = mapping.get("type")

    if mapping_type not in SUPPORTED_MAPPING_TYPES:
        raise ValueError(
            f"Unsupported mapping type: '{mapping_type}'."
        )


def load_dataset_config(
    name: str,
) -> dict[str, Any]:
    """
    Load and validate a dataset configuration from YAML.

    Args:
        name: Config filename without the .yaml extension.

    Returns:
        Parsed and validated dataset configuration.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the config structure is invalid.
    """

    config_path = CONFIG_DIR / f"{name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Dataset config must be a YAML object: "
            f"{config_path}"
        )

    validate_dataset_config(config)

    return config
