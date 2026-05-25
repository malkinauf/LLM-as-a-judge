import json
import logging
from pathlib import Path
from typing import Any
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
from datasets import concatenate_datasets

logger = logging.getLogger(__name__)


REQUIRED_DATASET_KEYS = {"id", "question", "model_response", "y_true"}

VALID_LABELS = {
    "truthful",
    "not_truthful",
    "safe",
    "unsafe"
}


def validate_dataset_entry(item: dict[str, Any], index: int) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"Dataset item {index} must be a dictionary.")

    missing_keys = REQUIRED_DATASET_KEYS - item.keys()
    if missing_keys:
        raise ValueError(
            f"Dataset item {index} is missing required keys: {missing_keys}"
        )

    if item["y_true"] not in VALID_LABELS:
        raise ValueError(
            f"Dataset item {index} has invalid label: {item['y_true']}"
        )

    if not isinstance(item["question"], str) or not item["question"].strip():
        raise ValueError(f"Dataset item {index} has empty question.")

    if not isinstance(item["model_response"], str) or not item["model_response"].strip():
        raise ValueError(f"Dataset item {index} has empty model_response.")


def validate_dataset(dataset: list[dict[str, Any]]) -> None:
    if not isinstance(dataset, list):
        raise ValueError("Dataset must be a list of dictionaries.")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty.")

    for i, item in enumerate(dataset):
        validate_dataset_entry(item, i)


def save_dataset_to_file(
    dataset: list[dict[str, Any]],
    path: str,
    overwrite: bool = False,
    raise_on_exists: bool = False
) -> None:
    """
    Save prepared dataset to a JSON file.
    
    Expected dataset format:
    [
        {
        "id": str,
        "question": str,
        "model_response":str,
        "y_true": str
        }
    ]"""

    validate_dataset(dataset)

    file_path = Path(path)

    if file_path.exists() and not overwrite:
        message = f"File already exists: {file_path}"

        if raise_on_exists:
            raise FileExistsError(message)
        
        logger.warning(message)
        logger.info("skipping save to avoid aoverwrite")
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved to {file_path}")


def prepare_beavertails(n_samples, seed=42):


    ds =load_dataset("PKU-Alignment/BeaverTails")["330k_train"]

    safe_ds = ds.filter(lambda x: x["is_safe"])
    unsafe_ds = ds.filter(lambda x:not x["is_safe"])
    
    half = n_samples // 2

    safe_ds =safe_ds.shuffle(seed=seed).select(range(half))
    unsafe_ds = unsafe_ds.shuffle(seed=seed).select(range(half))

    final_ds = concatenate_datasets([safe_ds,unsafe_ds])
    final_ds = final_ds.shuffle(seed=seed)

    data = []
                                            

    for i, row in enumerate(tqdm(final_ds)):
        label ="safe" if row["is_safe"] else "unsafe"

        data.append({
            "id": f"beavertails_{i}",
            "dataset":"beaver_tails",
            "question": row["prompt"],
            "model_response": row["response"],
            "y_true": label
        })

    return data


def prepare_truthfulqa(n_samples, seed=42):

    ds = load_dataset(
        "truthful_qa",
        "generation"
    )["validation"]

    ds = ds.shuffle(seed=seed).select(range(n_samples))

    data = []

    for i, row in enumerate(tqdm(ds)):

        question = row["question"]

        # TRUE sample
        if row["correct_answers"]:

            data.append({
                "id": f"truthfulqa_true_{i}",
                "dataset": "truthful_qa",
                "question": question,
                "model_response": row["correct_answers"][0],
                "y_true": "truthful",
            })

        # FALSE sample
        if row["incorrect_answers"]:

            data.append({
                "id": f"truthfulqa_false_{i}",
                "dataset": "truthful_qa",
                "question": question,
                "model_response": row["incorrect_answers"][0],
                "y_true": "not_truthful",
            })

    return data

def preview_dataset(data):

    labels = [x["y_true"] for x in data]

    print("Total samples:", len(data))
    print("Labels:", Counter(labels))

    print("\nExample:\n")
    print(json.dumps(data[0], indent=2, ensure_ascii=False))
                          


def load_dataset_from_file(path: str) -> list[dict[str, Any]]:
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
