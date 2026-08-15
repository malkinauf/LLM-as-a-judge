from evaluation_pipeline.prompts.criteria import HARMBENCH, SAFETY, TRUTHFULNESS


CRITERIA = {
    "truthfulness": TRUTHFULNESS,
    "safety": SAFETY,
    "harmbench": HARMBENCH,
}
