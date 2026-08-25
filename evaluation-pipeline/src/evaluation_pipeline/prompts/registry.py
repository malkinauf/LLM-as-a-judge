from evaluation_pipeline.prompts.loader import load_criterion

CRITERIA = {
    "truthfulness": load_criterion("truthfulness"),
    "safety": load_criterion("safety"),
    "correctness": load_criterion("correctness"),
    "harmbench": load_criterion("harmbench"),
    "hallucination": load_criterion("hallucination")
}