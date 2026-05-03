import json
from pathlib import Path

# ToDo Check fromat dataset, should be list of dicts with keys "question", "answer", "label"


def save_dataset_to_file(dataset, path: str, overwrite: bool = False) -> None:
    file_path = Path(path)

    if file_path.exists() and not overwrite:
        print(f"File already exists: {file_path}")
        print("Skipping save to avoid overwrite.")
        return

    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Dataset saved to {file_path}")


def load_dataset_from_file(path: str):
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    return dataset