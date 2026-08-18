from pathlib import Path
from typing import Any

import yaml


CRITERIA_DIR = Path(__file__).parent / "criteria"


def load_criterion(name: str) -> dict[str, Any]:
    """
    Load an evaluation criterion configuration from YAML.

    Args:
        name: Criterion name.

    Returns:
        Loaded criterion configuration.

    Raises:
        ValueError: If the criterion configuration does not exist.
    """

    path = CRITERIA_DIR / f"{name}.yaml"

    if not path.exists():
        raise ValueError(
            f"Unknown criterion: '{name}'."
        )

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)

SECOND_LEVEL_DIR = Path(__file__).parent / "second_level"


def load_second_level_review() -> dict[str, Any]:
    """
    Load the second-level review configuration from YAML.

    Returns:
        Second-level review configuration.
    """

    path = SECOND_LEVEL_DIR / "second_level_review.yaml"

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)