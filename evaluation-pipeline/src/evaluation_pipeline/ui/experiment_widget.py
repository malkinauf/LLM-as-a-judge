
"""AnyWidget model for the experiment dashboard.

Defines synchronized state and configuration for the experiment interface.
"""

from importlib.resources import files
from pathlib import Path

import anywidget
import traitlets

from evaluation_pipeline.experiment_config.model_registry import get_openrouter_models
from evaluation_pipeline.prompts.loader import load_criterion


def discover_prompt_variants():
    """Discover prompt variants and their availability by criterion."""
    criteria_dir = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "criteria"
    )

    variants = set()
    by_criterion = {}

    for path in criteria_dir.glob("*.yaml"):
        criterion = load_criterion(path.stem)
        available = sorted(criterion.get("descriptions", {}).keys())

        by_criterion[path.stem] = available
        variants.update(available)

    return {
        "baseline": sorted(variants),
        "dynamic": ["original", "paper_based"],
        "by_criterion": by_criterion,
    }



class ExperimentSetup(anywidget.AnyWidget):

    available_ollama_models = traitlets.List(
        traitlets.Unicode(), default_value=[]
    ).tag(sync=True)

    available_openrouter_models = traitlets.List(
        traitlets.Dict(), default_value=[]
    ).tag(sync=True)

    available_prompt_variants = traitlets.Dict(
        default_value={}
    ).tag(sync=True)

    selected_backend = traitlets.Unicode(
        default_value="ollama"
    ).tag(sync=True)

    selected_models = traitlets.List(
        traitlets.Dict(), default_value=[]
    ).tag(sync=True)

    available_datasets = traitlets.List(
        traitlets.Dict(), default_value=[]
    ).tag(sync=True)

    selected_datasets = traitlets.List(
        traitlets.Dict(), default_value=[]
    ).tag(sync=True)

    selected_methods = traitlets.List(
        traitlets.Dict(), default_value=[]
    ).tag(sync=True)

    confidence_mode = traitlets.Bool(default_value=False).tag(sync=True)
    use_wandb = traitlets.Bool(default_value=False).tag(sync=True)
    launch_request = traitlets.Int(default_value=0).tag(sync=True)
    stop_request = traitlets.Int(default_value=0).tag(sync=True)
    run_status = traitlets.Unicode(default_value="Ready").tag(sync=True)
    run_current = traitlets.Int(default_value=0).tag(sync=True)
    run_total = traitlets.Int(default_value=0).tag(sync=True)
    run_active = traitlets.Bool(default_value=False).tag(sync=True)
    current_run_label = traitlets.Unicode(default_value="").tag(sync=True)
    run_results = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
    current_run_started_at = traitlets.Float(default_value=0.0).tag(sync=True)
    execution_busy = traitlets.Bool(default_value=False).tag(sync=True)
    sample_current = traitlets.Int(default_value=0).tag(sync=True)
    sample_total = traitlets.Int(default_value=0).tag(sync=True)
    sample_last_completed_at = traitlets.Float(default_value=0.0).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.available_openrouter_models = get_openrouter_models()
        self.available_prompt_variants = discover_prompt_variants()

    _esm = files("evaluation_pipeline").joinpath(
        "ui", "experiment.js"
    ).read_text(encoding="utf-8")

    _css = files("evaluation_pipeline").joinpath(
        "ui", "experiment.css"
    ).read_text(encoding="utf-8")
