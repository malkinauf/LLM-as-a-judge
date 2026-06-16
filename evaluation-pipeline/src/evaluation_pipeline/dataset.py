import random

import requests
import json
import logging
from pathlib import Path
from typing import Any
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
from collections import Counter


logger = logging.getLogger(__name__)

REQUIRED_DATASET_KEYS = {"id", "dataset",
                         "question", "model_response", "y_true"}

VALID_LABELS = {
    "truthful",
    "not_truthful",
    "safe",
    "not_safe",
    "attack_successful",
    "attack_failed"
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


def save_dataset_to_file(
    dataset: list[dict[str, Any]],
    path: str | Path,
    overwrite: bool = False,
    raise_on_exists: bool = False,
) -> None:
    """
    Save a validated dataset to a JSON file.

    Args:
        dataset: Dataset to save.
        path: Path to the output JSON file. Accepts either a string
            or a pathlib.Path object.
        overwrite: Whether to overwrite an existing file.
        raise_on_exists: Raise FileExistsError instead of
            skipping save when the file already exists.

    Raises:
        ValueError: If the dataset is invalid.
        FileExistsError: If the file exists and
            raise_on_exists is True.
    """

    validate_dataset(dataset)
    file_path = Path(path)

    if file_path.exists() and not overwrite:
        message = f"File already exists: {file_path}"
        if raise_on_exists:
            raise FileExistsError(message)
        logger.warning(message)
        logger.info("Skipping save to avoid overwriting existing file.")
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved to {file_path}")


def prepare_truthfulqa(n_samples: int, seed: int = 42) -> list[dict[str, Any]]:
    """
    Prepare TruthfulQA dataset.

    Args:
        n_samples: Desired number of output samples.
            Must be a positive even number.
        seed: Random seed for shuffling.

    Returns:
        Prepared dataset entries.
    """

    if n_samples <= 0 or n_samples % 2 != 0:
        raise ValueError("n_samples must be a positive even number.")

    n_questions = n_samples // 2
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    if n_questions > len(dataset):
        raise ValueError(
            f"Requested {n_samples} samples, but TruthfulQA validation split "
            f"contains only {len(dataset)} questions. "
            f"Maximum possible n_samples is {len(dataset) * 2}."
        )

    dataset = dataset.shuffle(seed=seed).select(range(n_questions))

    data = []
    entry_id = 0

    for row in tqdm(dataset, desc="Preparing TruthfulQA"):
        question = row["question"]

        if row["correct_answers"]:
            data.append(
                {
                    "id": f"truthfulqa_{entry_id}",
                    "dataset": "truthful_qa",
                    "question": question,
                    "model_response": row["correct_answers"][0],
                    "y_true": "truthful",
                }
            )
            entry_id += 1

        if row["incorrect_answers"]:
            data.append(
                {
                    "id": f"truthfulqa_{entry_id}",
                    "dataset": "truthful_qa",
                    "question": question,
                    "model_response": row["incorrect_answers"][0],
                    "y_true": "not_truthful",
                }
            )
            entry_id += 1
    return data


def prepare_beavertails(
    n_samples: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Prepare a balanced BeaverTails dataset for evaluation.

    The output contains the same number of safe and not_safe samples.
    Therefore, n_samples must be a positive even number.

    Args:
        n_samples: Desired number of output samples.
        seed: Random seed used for dataset shuffling.

    Returns:
        List of prepared dataset entries.

    Raises:
        ValueError: If n_samples is not a positive even number.
    """

    if n_samples <= 0 or n_samples % 2 != 0:
        raise ValueError("n_samples must be a positive even number.")

    dataset = load_dataset("PKU-Alignment/BeaverTails")["330k_train"]

    safe_dataset = dataset.filter(lambda x: x["is_safe"])
    not_safe_dataset = dataset.filter(lambda x: not x["is_safe"])

    half = n_samples // 2

    max_samples = min(len(safe_dataset), len(not_safe_dataset)) * 2

    if n_samples > max_samples:
        raise ValueError(
            f"Requested {n_samples} samples, but the maximum "
            f"balanced BeaverTails dataset size is {max_samples}."
        )

    safe_dataset = safe_dataset.shuffle(seed=seed).select(range(half))
    not_safe_dataset = not_safe_dataset.shuffle(seed=seed).select(range(half))

    final_dataset = concatenate_datasets([safe_dataset, not_safe_dataset])
    final_dataset = final_dataset.shuffle(seed=seed)

    data = []

    for i, row in enumerate(tqdm(final_dataset, desc="Preparing BeaverTails")):
        label = "safe" if row["is_safe"] else "not_safe"

        data.append({
            "id": f"beavertails_{i}",
            "dataset": "beaver_tails",
            "question": row["prompt"],
            "model_response": row["response"],
            "y_true": label,
        })

    return data


def preview_dataset(data: list[dict[str, Any]]) -> None:
    """
    Print a short summary and one example from the dataset.

    Args:
        data: Dataset entries to preview.
    """

    validate_dataset(data)

    labels = [x["y_true"] for x in data]

    print("Total samples:", len(data))
    print("Labels:", Counter(labels))

    print("\nExample:\n")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))


def load_dataset_from_file(path: str) -> list[dict[str, Any]]:
    """
    Load and validate a dataset from a JSON file.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        Validated dataset entries.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or
            does not match the expected dataset format.
    """

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as f:
            dataset = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in dataset file: {file_path}"
        ) from e

    validate_dataset(dataset)

    logger.info(f"Dataset loaded from {file_path}")

    return dataset


def get_harmbench_classifier_val_set(
    path: str | Path = "data/classifier_val_sets/text_behaviors_val_set.json",
    force_download: bool = False,
) -> Path:
    file_path = Path(path)

    url = (
        "https://raw.githubusercontent.com/"
        "centerforaisafety/HarmBench/main/"
        "data/classifier_val_sets/text_behaviors_val_set.json"
    )

    if file_path.exists() and not force_download:
        return file_path

    file_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    text = response.text.strip()

    if not text.startswith("{"):
        raise ValueError(
            "Downloaded file does not look like JSON. "
            f"First 200 chars:\n{text[:200]}"
        )

    file_path.write_text(text, encoding="utf-8")

    return file_path


def prepare_harmbench(
        n_samples: int | None = None,
        seed: int = 42,
        include_disputed: bool = True,) -> list[dict[str, Any]]:
    """
    Prepare HarmBench dataset.

    Args:
        include_disputed: If False, keep only examples where
            all human annotators agree: 0/0/0 or 1/1/1.

    Returns:
        List of prepared dataset entries with schema:
        id, dataset, question, model_response, y_true.
    """
    if n_samples is not None and (n_samples <= 0 or n_samples % 2 != 0):
        raise ValueError("n_samples must be a positive even number.")

    path = get_harmbench_classifier_val_set()

    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    successful = []
    failed = []
    entry_id = 0

    for behavior_id, examples in tqdm(raw_data.items(), desc="Preparing HarmBench"):
        if not isinstance(examples, list):
            continue

        for example in examples:
            human_labels = [
                example.get("human_0"),
                example.get("human_1"),
                example.get("human_2"),
            ]

            if any(label not in {0, 1} for label in human_labels):
                continue

            votes = sum(human_labels)

            if not include_disputed and votes not in {0, 3}:
                continue

            y_true = "attack_successful" if votes >= 2 else "attack_failed"

            test_case = example.get("test_case", "")
            generation = example.get("generation", "")

            if not isinstance(test_case, str) or not test_case.strip():
                continue

            if not isinstance(generation, str) or not generation.strip():
                continue

            item = {
                "id": f"harmbench_{entry_id}",
                "dataset": "harmbench_classifier_val",
                "question": test_case,
                "model_response": generation,
                "y_true": y_true,
            }

            if y_true == "attack_successful":
                successful.append(item)
            else:
                failed.append(item)

            entry_id += 1

    if n_samples is None:
        data = successful + failed
        random.Random(seed).shuffle(data)
        validate_dataset(data)
        return data

    half = n_samples // 2

    if half > len(successful) or half > len(failed):
        raise ValueError(
            f"Requested {n_samples} balanced samples, but available: "
            f"{len(successful)} successful and {len(failed)} failed. "
            f"Maximum balanced n_samples is {min(len(successful), len(failed)) * 2}."
        )

    rng = random.Random(seed)

    data = (
        rng.sample(successful, half)
        + rng.sample(failed, half)
    )

    rng.shuffle(data)

    for i, item in enumerate(data):
        item["id"] = f"harmbench_{i}"

    validate_dataset(data)
    return data
