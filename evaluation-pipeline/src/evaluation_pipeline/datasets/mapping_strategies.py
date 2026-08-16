from abc import ABC, abstractmethod
from typing import Any


def _resolve_value(
    row: dict[str, Any],
    field_config: dict[str, Any],
) -> Any:
    """
    Resolve a field value from a dataset row or config.
    """

    if "source" in field_config:
        source = field_config["source"]

        if source not in row:
            raise ValueError(
                f"Source field '{source}' not found in dataset row."
            )

        value = row[source]

    elif "value" in field_config:
        value = field_config["value"]

    else:
        raise ValueError(
            "Field config must define 'source' or 'value'."
        )

    if "index" in field_config:
        index = field_config["index"]
        value = value[index]

    if "include" in field_config:
        for field in field_config["include"]:
            extra = row[field]

            if isinstance(extra, list):
                extra = "\n".join(
                    f"{chr(65 + i)}. {item}"
                    for i, item in enumerate(extra)
                )

            value = f"{value}\n\n{extra}"

    if "values" in field_config:
        value = field_config["values"][value]

    return value


class MappingStrategy(ABC):
    """
    Base class for dataset mapping strategies.
    """

    @abstractmethod
    def map(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        pass


class SingleMapping(MappingStrategy):
    """
    Map one raw row to one canonical sample.
    """

    def map(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "question": _resolve_value(
                    row,
                    config["question"],
                ),
                "model_response": _resolve_value(
                    row,
                    config["model_response"],
                ),
                "y_true": _resolve_value(
                    row,
                    config["y_true"],
                ),
            }
        ]


class MultiMapping(MappingStrategy):
    """
    Map one raw row to multiple configured samples.
    """

    def map(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        samples = []

        for sample_config in config["samples"]:
            samples.append(
                {
                    "question": _resolve_value(
                        row,
                        sample_config["question"],
                    ),
                    "model_response": _resolve_value(
                        row,
                        sample_config["model_response"],
                    ),
                    "y_true": _resolve_value(
                        row,
                        sample_config["y_true"],
                    ),
                }
            )

        return samples


class AlignedMapping(MappingStrategy):
    """
    Map multiple responses to their aligned labels.
    """

    def map(
        self,
        row: dict[str, Any],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        question = _resolve_value(
            row,
            config["question"],
        )

        response_prefix = config["responses"]["prefix"]

        label_config = config["labels"]
        labels = row[label_config["source"]]

        samples = []

        for index, label in enumerate(
            labels,
            start=1,
        ):
            response = row.get(
                f"{response_prefix}{index}"
            )

            if not response:
                continue

            if "values" in label_config:
                label = label_config["values"][label]

            samples.append(
                {
                    "question": question,
                    "model_response": response,
                    "y_true": label,
                }
            )

        return samples


MAPPING_STRATEGIES = {
    "single": SingleMapping(),
    "multi": MultiMapping(),
    "aligned": AlignedMapping(),
}


def get_mapping_strategy(
    mapping_type: str,
) -> MappingStrategy:
    """
    Return the configured mapping strategy.
    """

    if mapping_type not in MAPPING_STRATEGIES:
        raise ValueError(
            f"Unsupported mapping type: '{mapping_type}'"
        )

    return MAPPING_STRATEGIES[mapping_type]
