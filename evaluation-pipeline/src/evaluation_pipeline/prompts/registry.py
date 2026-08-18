from evaluation_pipeline.prompts.loader import load_criterion
from evaluation_pipeline.prompts.loader import (
    load_criterion,
    load_second_level_review,
)


CRITERIA = {
    "truthfulness": load_criterion("truthfulness"),
    "safety": load_criterion("safety"),
    "correctness": load_criterion("correctness"),
    "harmbench": load_criterion("harmbench"),
}
SECOND_LEVEL_REVIEW = load_second_level_review()