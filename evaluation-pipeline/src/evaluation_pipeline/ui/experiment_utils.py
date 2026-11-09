
"""Dataset/model discovery and experiment plan construction for the Dashboard."""

from itertools import product
from evaluation_pipeline.datasets.config_loader import load_dataset_config
from evaluation_pipeline.prompts.loader import load_criterion


def discover_prepared_datasets():
    from evaluation_pipeline.datasets.dataset_io import _get_project_root
    from evaluation_pipeline.datasets.config_loader import CONFIG_DIR, load_dataset_config

    prepared_dir = _get_project_root() / "datasets" / "prepared"
    if not prepared_dir.is_dir():
        return []

    names = sorted((p.stem for p in CONFIG_DIR.glob("*.yaml")), key=len, reverse=True)
    datasets = []
    for path in sorted(prepared_dir.glob("*.json")):
        for name in names:
            prefix = f"{name}_"
            if not path.stem.startswith(prefix):
                continue
            variant = path.stem[len(prefix):]
            if not variant:
                continue
            try:
                load_dataset_config(name)
            except (ValueError, FileNotFoundError):
                continue
            datasets.append({
                "name": name,
                "variant": variant,
                "dataset_id": path.stem,
                "evaluation_criterion": load_dataset_config(name)["evaluation_criterion"],
            })
            break
    return datasets


def discover_ollama_models():
    """Return installed Ollama model IDs, or [] if Ollama is unavailable."""
    try:
        from ollama import list as ollama_list
        response = ollama_list()
        return sorted({item.model for item in response.models if item.model})
    except Exception as exc:
        print(f"Ollama model discovery unavailable: {exc}")
        return []


def build_plan_with_skipped(ui):
    models = list(ui.selected_models)
    datasets = list(ui.selected_datasets)
    methods = list(ui.selected_methods)
    backend = ui.selected_backend

    if backend not in {"ollama", "openrouter"}:
        raise ValueError(f"Unsupported backend: {backend}")
    if not all((models, datasets, methods)):
        raise ValueError("Select at least one model, dataset and method.")
    if any(m["backend"] != backend for m in models):
        raise ValueError("Selected models must match the active backend.")

    available = {d["dataset_id"] for d in ui.available_datasets}
    if any(d["dataset_id"] not in available for d in datasets):
        raise ValueError("Selected dataset is no longer available.")

    plan = []
    skipped = []
    criterion_cache = {}

    for m, d, method in product(models, datasets, methods):
        method_name = method["method"]

        if method_name not in {"baseline", "second_level", "dynamic"}:
            raise ValueError(f"Unknown method: {method_name}")

        detail = method["baseline_criterion_detail"]
        if not isinstance(detail, str) or not detail:
            raise ValueError("Baseline criterion detail must be a non-empty string.")

        prediction = (
            method["dynamic_prediction_variant"]
            if method_name == "dynamic"
            else "original"
        )

        if method_name == "dynamic" and not isinstance(prediction, str):
            raise ValueError("Dynamic prediction variant must be a string.")

        second_level_variant = (
            method.get("second_level_variant", "original")
            if method_name == "second_level"
            else "original"
        )
        if second_level_variant not in {"original", "adhd"}:
            raise ValueError(
                f"Unknown second-level prompt variant: {second_level_variant}"
            )

        dataset_config = load_dataset_config(d["name"])
        criterion = dataset_config["evaluation_criterion"]

        if criterion not in criterion_cache:
            criterion_cache[criterion] = load_criterion(criterion)

        descriptions = criterion_cache[criterion].get("descriptions", {})

        if method_name == "dynamic":
            prediction_instructions = (
                criterion_cache[criterion]
                .get("dynamic", {})
                .get("prediction_instructions", {})
            )

            if prediction not in prediction_instructions:
                skipped.append({
                    "model": m["model"],
                    "dataset_id": d["dataset_id"],
                    "method": method_name,
                    "baseline_criterion_detail": detail,
                    "dynamic_prediction_variant": prediction,
                    "reason": (
                        f"Dynamic prediction variant '{prediction}' "
                        f"is not defined for criterion '{criterion}'."
                    ),
                })
                continue


        if detail not in descriptions:
            skipped.append({
                "model": m["model"],
                "dataset_id": d["dataset_id"],
                "method": method_name,
                "baseline_criterion_detail": detail,
                "reason": (
                    f"Baseline variant '{detail}' is not defined "
                    f"for criterion '{criterion}'."
                ),
            })
            continue

        plan.append({
            "backend": backend,
            "model": m["model"],
            "dataset_id": d["dataset_id"],
            "dataset_name": d["name"],
            "method": method_name,
            "evaluation_criterion": criterion,
            "baseline_criterion_detail": detail,
            "dynamic_prediction_variant": prediction,
            "second_level_variant": second_level_variant,
            "confidence_mode": bool(ui.confidence_mode),
            "use_wandb": bool(ui.use_wandb),
        })

    return plan, skipped


def build_plan(ui):
    """Return only runnable combinations; keep the existing caller API."""
    plan, _skipped = build_plan_with_skipped(ui)
    return plan
