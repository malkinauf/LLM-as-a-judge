from evaluation_pipeline.prompts.criteria import CORRECTNESS, HARMBENCH, SAFETY, TRUTHFULNESS


CRITERIA = {
    "truthfulness": TRUTHFULNESS,
    "safety": SAFETY,
    "harmbench": HARMBENCH,
    "correctness": CORRECTNESS,
}
