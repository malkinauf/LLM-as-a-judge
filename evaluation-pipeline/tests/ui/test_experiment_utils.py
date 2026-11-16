
from types import SimpleNamespace

from evaluation_pipeline.ui import experiment_utils


def test_build_plan_skips_unsupported_dynamic(monkeypatch):
    monkeypatch.setattr(
        experiment_utils,
        "load_dataset_config",
        lambda name: {"evaluation_criterion": "correctness"},
    )

    monkeypatch.setattr(
        experiment_utils,
        "load_criterion",
        lambda name: {
            "descriptions": {"basic": "Basic criterion"},
            "dynamic": {
                "prediction_instructions": {
                    "paper_based": "Prediction instructions"
                }
            },
        },
    )

    dataset = {
        "name": "test_dataset",
        "dataset_id": "test_dataset_4",
    }

    ui = SimpleNamespace(
        selected_backend="ollama",
        selected_models=[
            {"backend": "ollama", "model": "test_model"}
        ],
        selected_datasets=[dataset],
        available_datasets=[dataset],
        selected_methods=[
            {
                "method": "dynamic",
                "baseline_criterion_detail": "basic",
                "dynamic_prediction_variant": "minimal",
            }
        ],
        confidence_mode=False,
        use_wandb=False,
    )

    plan, skipped = experiment_utils.build_plan_with_skipped(ui)

    assert len(plan) == 0
    assert len(skipped) == 1
    assert skipped[0]["dynamic_prediction_variant"] == "minimal"
    assert "not defined" in skipped[0]["reason"]


def test_build_plan_accepts_supported_dynamic(monkeypatch):
    monkeypatch.setattr(
        experiment_utils,
        "load_dataset_config",
        lambda name: {"evaluation_criterion": "correctness"},
    )

    monkeypatch.setattr(
        experiment_utils,
        "load_criterion",
        lambda name: {
            "descriptions": {"basic": "Basic criterion"},
            "dynamic": {
                "prediction_instructions": {
                    "paper_based": "Prediction instructions"
                }
            },
        },
    )

    dataset = {
        "name": "test_dataset",
        "dataset_id": "test_dataset_4",
    }

    ui = SimpleNamespace(
        selected_backend="ollama",
        selected_models=[
            {"backend": "ollama", "model": "test_model"}
        ],
        selected_datasets=[dataset],
        available_datasets=[dataset],
        selected_methods=[
            {
                "method": "dynamic",
                "baseline_criterion_detail": "basic",
                "dynamic_prediction_variant": "paper_based",
            }
        ],
        confidence_mode=False,
        use_wandb=False,
    )

    plan, skipped = experiment_utils.build_plan_with_skipped(ui)

    assert len(plan) == 1
    assert len(skipped) == 0
    assert plan[0]["dynamic_prediction_variant"] == "paper_based"
    assert plan[0]["evaluation_criterion"] == "correctness"
