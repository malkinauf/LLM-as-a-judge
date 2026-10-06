import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay,
)


CRITERION_LABELS = {
    "truthfulness": {"positive": "truthful", "negative": "not_truthful"},
    "safety": {"positive": "not_safe", "negative": "safe"},
    "harmbench": {"positive": "attack_success", "negative": "attack_failed"},
    "correctness": {"positive": "correct", "negative": "incorrect"},
    "hallucination": {"positive": "hallucinated", "negative": "not_hallucinated"},
}


def compute_metrics(
    df_results,
    run_id,
    method,
    model,
    dataset_id,
    baseline_criterion,
    baseline_criterion_detail,
    results_dir=None,
):
    total_samples = len(df_results)
    if total_samples == 0:
        raise ValueError("No experiment results available.")

    label_config = CRITERION_LABELS[baseline_criterion]
    positive_label = label_config["positive"]
    negative_label = label_config["negative"]
    labels = [positive_label, negative_label]

    invalid_true_labels = df_results[~df_results["true_label"].isin(labels)]
    if len(invalid_true_labels) > 0:
        raise ValueError(
            "Dataset contains unexpected true labels: "
            f"{sorted(invalid_true_labels['true_label'].unique())}"
        )

    parsing_errors = df_results[df_results["predicted_label"] == "parsing_error"]
    runtime_errors = df_results[df_results["predicted_label"] == "runtime_error"]
    invalid_labels = df_results[
        (~df_results["predicted_label"].isin(labels))
        & (~df_results["predicted_label"].isin(["parsing_error", "runtime_error"]))
    ]

    valid_results = df_results[df_results["predicted_label"].isin(labels)].copy()
    valid_samples = len(valid_results)
    invalid_samples = total_samples - valid_samples
    coverage = valid_samples / total_samples if total_samples > 0 else 0

    print("Output Quality")
    print("=" * 40)
    print(f"Total samples: {total_samples}")
    print(f"Valid outputs: {valid_samples} ({coverage:.2%})")
    print(f"Parsing errors: {len(parsing_errors)}")
    print(f"Runtime errors: {len(runtime_errors)}")
    print(f"Invalid labels: {len(invalid_labels)}")

    if valid_samples == 0:
        raise ValueError("No valid predictions available for metric calculation.")

    y_true = valid_results["true_label"]
    y_pred = valid_results["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm_labels = [negative_label, positive_label]
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)
    tn, fp, fn, tp = cm.ravel()

    print("\nClassification Metrics")
    print("=" * 40)
    print("Positive label:", positive_label)
    print("Negative label:", negative_label)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 (positive): {f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"MCC: {mcc:.4f}")
    print(f"Coverage: {coverage:.2%}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    fig, ax = plt.subplots(figsize=(7, 7))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format="d")
    for text in disp.text_.flatten():
        text.set_fontsize(24)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlabel("Predicted label", fontsize=16)
    ax.set_ylabel("True label", fontsize=16)
    ax.set_title(
        f"{baseline_criterion} Confusion Matrix:\n{run_id} - {model} - {method}",
        fontsize=18,
        pad=15,
    )
    plt.tight_layout()
    plt.show()

    summary_metrics = {
        "run_id": run_id,
        "method": method,
        "model": model,
        "dataset_id": dataset_id,
        "baseline_criterion": baseline_criterion,
        "baseline_criterion_detail": baseline_criterion_detail,
        "positive_label": positive_label,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": invalid_samples,
        "coverage": coverage,
        "parsing_errors": len(parsing_errors),
        "runtime_errors": len(runtime_errors),
        "invalid_labels": len(invalid_labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "cohen_kappa": kappa,
        "mcc": mcc,
    }

    # ============================================
    # Second-Level analysis
    # ============================================
    second_level_metrics = {}

    if method == "second_level":
        df_sl = df_results.copy()
        n_total = len(df_sl)

        first_correct = df_sl["first_level_label"] == df_sl["true_label"]
        final_correct = df_sl["predicted_label"] == df_sl["true_label"]

        first_level_accuracy = first_correct.mean() if n_total > 0 else 0
        final_accuracy = final_correct.mean() if n_total > 0 else 0
        accuracy_delta = final_accuracy - first_level_accuracy

        valid_second_level = df_sl[
            df_sl["second_level_verdict"].isin(["correct", "not_correct"])
        ].copy()

        second_level_coverage = (
            len(valid_second_level) / n_total if n_total > 0 else 0
        )

        valid_classification = valid_second_level[
            ~valid_second_level["predicted_label"].isin(["parsing_error", "runtime_error"])
            & valid_second_level["first_level_label"].notna()
            & valid_second_level["predicted_label"].notna()
            & valid_second_level["true_label"].notna()
        ].copy()

        n_valid_classification = len(valid_classification)
        classification_coverage = (
            n_valid_classification / n_total if n_total > 0 else 0
        )

        effect_first_correct = (
            valid_classification["first_level_label"]
            == valid_classification["true_label"]
        )
        effect_final_correct = (
            valid_classification["predicted_label"]
            == valid_classification["true_label"]
        )

        corrected = (~effect_first_correct) & effect_final_correct
        degraded = effect_first_correct & (~effect_final_correct)
        unchanged_correct = effect_first_correct & effect_final_correct
        unchanged_wrong = (~effect_first_correct) & (~effect_final_correct)

        n_first_correct = int(effect_first_correct.sum())
        n_first_wrong = int((~effect_first_correct).sum())
        corrected_count = int(corrected.sum())
        degraded_count = int(degraded.sum())
        net_gain_count = corrected_count - degraded_count

        label_changed = (
            valid_classification["predicted_label"]
            != valid_classification["first_level_label"]
        )
        label_change_count = int(label_changed.sum())
        label_change_rate = (
            label_changed.mean() if n_valid_classification > 0 else 0
        )

        correction_rate = (
            corrected_count / n_first_wrong if n_first_wrong > 0 else 0
        )
        degradation_rate = (
            degraded_count / n_first_correct if n_first_correct > 0 else 0
        )

        second_level_metrics = {
            "first_level_accuracy": first_level_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_delta": accuracy_delta,
            "corrected_count": corrected_count,
            "degraded_count": degraded_count,
            "net_gain_count": net_gain_count,
            "correction_rate": correction_rate,
            "degradation_rate": degradation_rate,
            "label_change_count": label_change_count,
            "label_change_rate": label_change_rate,
            "second_level_coverage": second_level_coverage,
            "classification_coverage": classification_coverage,
        }

        print("\nSecond-Level Analysis")
        print("=" * 40)
        print(f"First-level accuracy: {first_level_accuracy:.2%}")
        print(f"Final accuracy: {final_accuracy:.2%}")
        print(f"Accuracy delta: {accuracy_delta:+.2%}")
        print("\nEffect:")
        print(f"Corrected mistakes: {corrected_count}")
        print(f"Degraded predictions: {degraded_count}")
        print(f"Unchanged correct: {int(unchanged_correct.sum())}")
        print(f"Unchanged wrong: {int(unchanged_wrong.sum())}")
        print(f"Net gain: {net_gain_count}")
        print("\nRates:")
        print(f"Correction rate: {correction_rate:.2%}")
        print(f"Degradation rate: {degradation_rate:.2%}")
        print(f"Label change rate: {label_change_rate:.2%}")
        print(f"Second-level coverage: {second_level_coverage:.2%}")
        print(f"Classification coverage: {classification_coverage:.2%}")

    # ============================================
    # Dynamic analysis
    # ============================================
    dynamic_metrics = {}

    if method == "dynamic":
        if "prediction_raw_output" in df_results.columns:
            has_prediction = (
                df_results["prediction_raw_output"].notna()
                & (df_results["prediction_raw_output"].astype(str).str.strip() != "")
            )
        else:
            has_prediction = pd.Series(False, index=df_results.index)

        prediction_available_count = int(has_prediction.sum())
        prediction_output_coverage = (
            has_prediction.mean() if len(df_results) > 0 else 0
        )

        dynamic_metrics = {
            "prediction_available_count": prediction_available_count,
            "prediction_coverage": prediction_output_coverage,
        }

        print("\nDynamic Analysis")
        print("=" * 40)
        print(f"Prediction available: {prediction_available_count}")
        print(f"Prediction coverage: {prediction_output_coverage:.2%}")

    # ============================================
    # Confidence-Verteilung in summary_metrics
    # ============================================
    if "first_level_confidence" in df_results.columns:
        conf = pd.to_numeric(df_results["first_level_confidence"], errors="coerce")
        conf_counts = conf.value_counts(normalize=True).sort_index()

        for level in [1, 2, 3, 4, 5]:
            summary_metrics[f"confidence_{level}_pct"] = round(
                float(conf_counts.get(level, 0.0)) * 100, 1
            )

        summary_metrics["confidence_mean"] = round(float(conf.mean()), 2)

    summary_metrics.update(second_level_metrics)
    summary_metrics.update(dynamic_metrics)
    df_summary_metrics = pd.DataFrame([summary_metrics])

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        safe_model = model.replace(":", "_").replace("/", "_")
        base_name = f"{run_id}_{method}_{safe_model}_{dataset_id}"

        results_path = results_dir / f"{base_name}_results.csv"
        summary_path = results_dir / f"{base_name}_summary.csv"

        df_results.to_csv(results_path, index=False)
        df_summary_metrics.to_csv(summary_path, index=False)

        print("\nSaved experiment results:")
        print(results_path)
        print(summary_path)

    return {
        "summary_metrics": summary_metrics,
        "df_summary_metrics": df_summary_metrics,
        "second_level_metrics": second_level_metrics,
        "dynamic_metrics": dynamic_metrics,
        "fig": fig,
        "positive_label": positive_label,
        "negative_label": negative_label,
    }