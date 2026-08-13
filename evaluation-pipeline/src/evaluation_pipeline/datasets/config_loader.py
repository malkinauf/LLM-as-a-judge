from pathlib import Path
from typing import Any

import yaml


CONFIG_DIR = Path(__file__).parent / "configs"


REQUIRED_CONFIG_KEYS = {
    "name",
    "output_dataset_name",
    "source",
    "mapping",
    "label",
    "sampling",
}


def validate_dataset_config(config: dict[str, Any]) -> None:
    missing_keys = REQUIRED_CONFIG_KEYS - set(config.keys())

    if missing_keys:
        raise ValueError(
            f"Dataset config is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    source = config["source"]

    if not isinstance(source, dict):
        raise ValueError("'source' must be a mapping.")

    for key in ("type", "dataset", "split"):
        if key not in source:
            raise ValueError(
                f"Dataset config source is missing '{key}'."
            )

    mapping = config["mapping"]

    if not isinstance(mapping, dict):
        raise ValueError("'mapping' must be a mapping.")

    for key in ("question", "model_response"):
        if key not in mapping:
            raise ValueError(
                f"Dataset config mapping is missing '{key}'."
            )

    label = config["label"]

    if not isinstance(label, dict):
        raise ValueError("'label' must be a mapping.")

    for key in ("source", "values"):
        if key not in label:
            raise ValueError(
                f"Dataset config label is missing '{key}'."
            )


def load_dataset_config(name: str) -> dict[str, Any]:
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
