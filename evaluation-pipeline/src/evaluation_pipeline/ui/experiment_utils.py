"""Dataset/model discovery and experiment plan construction for the Dashboard."""

from itertools import product
from evaluation_pipeline.datasets.config_loader import load_dataset_config


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
            datasets.append({"name": name, "variant": variant, "dataset_id": path.stem})
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


def build_plan(ui):
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
    for m, d, method in product(models, datasets, methods):
        method_name = method["method"]
        if method_name not in {"baseline", "second_level", "dynamic"}:
            raise ValueError(f"Unknown method: {method_name}")
        detail = method["baseline_criterion_detail"]
        if detail not in {"basic", "detailed"}:
            raise ValueError(f"Unknown criterion detail: {detail}")
        prediction = (
            method["dynamic_prediction_variant"] if method_name == "dynamic"
            else "original"
        )
        if method_name == "dynamic" and prediction not in {"original", "paper_based"}:
            raise ValueError(f"Unknown dynamic variant: {prediction}")
        dataset_config = load_dataset_config(d["name"])
        plan.append({
            "backend": backend,
            "model": m["model"],
            "dataset_id": d["dataset_id"],
            "dataset_name": d["name"],
            "method": method_name,
            "evaluation_criterion": dataset_config["evaluation_criterion"],
            "baseline_criterion_detail": detail,
            "dynamic_prediction_variant": prediction,
            "confidence_mode": bool(ui.confidence_mode),
            "use_wandb": bool(ui.use_wandb),
        })
    return plan

