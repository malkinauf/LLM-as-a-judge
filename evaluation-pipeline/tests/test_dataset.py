import json
import tempfile
import unittest
from pathlib import Path

from src.dataset import load_dataset_from_file


class TestDataset(unittest.TestCase):
    def test_load_empty_dataset_returns_empty_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text("[]", encoding="utf-8")

            data = load_dataset_from_file(str(path))

        self.assertEqual(data, [])

    def test_load_dataset_missing_fields_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps([{"id": "1", "question": "q", "answer": "a"}]), encoding="utf-8")

            with self.assertRaises(ValueError):
                load_dataset_from_file(str(path))

    def test_load_dataset_valid_json(self) -> None:
        payload = [{"id": "1", "question": "q", "answer": "a", "y_true": "truthful"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            loaded = load_dataset_from_file(str(path))

        self.assertEqual(loaded, payload)


if __name__ == "__main__":
    unittest.main()
