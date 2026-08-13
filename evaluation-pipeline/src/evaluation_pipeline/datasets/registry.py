
from evaluation_pipeline.dataset import prepare_beavertails, prepare_truthfulqa
from evaluation_pipeline.tmp_dataset import validate_dataset


DATASET_PREPARERS = {
    "truthfulqa": prepare_truthfulqa,
    "beavertails": prepare_beavertails,
}


def prepare_dataset(
    name: str,
    n_samples: int,
):
    if name not in DATASET_PREPARERS:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available datasets: {list(DATASET_PREPARERS)}"
        )

    preparer = DATASET_PREPARERS[name]

    dataset = preparer(n_samples=n_samples)

    validate_dataset(dataset)
    return dataset
