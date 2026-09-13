from typing import Any, Dict, List, Tuple
import json
import math
import random

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from scipy.stats import chi2


TASK_LABELS = {
    "truthfulness": ["truthful", "not_truthful"],
    "helpfulness": ["helpful", "not_helpful"],
    "toxicity": ["toxic", "not_toxic"],
    "safety": ["safe", "unsafe"],
}


def _valid_prediction_rows(
    results: List[Dict[str, Any]],
    labels: List[str],
) -> List[Dict[str, Any]]:
    return [r for r in results if r.get("predicted_label") in labels]


def _classification_arrays(
    results: List[Dict[str, Any]],
    labels: List[str],
) -> Tuple[List[Any], List[Any]]:
    valid_results = _valid_prediction_rows(results, labels)
    y_true = [r["true_label"] for r in valid_results]
    y_pred = [r["predicted_label"] for r in valid_results]
    return y_true, y_pred


def compute_classification_metrics(results: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    y_true, y_pred = _classification_arrays(results, labels)

    if not y_true:
        return {"error": "no valid predictions"}

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    return metrics


def build_summary_metrics(
    results: List[Dict[str, Any]],
    labels: List[str],
    *,
    run_id: str,
    method: str,
    model: str,
    task_type: str,
    dataset_file: str,
    baseline_prompt_file: str,
    second_level_prompt_file: str = "",
) -> Dict[str, Any]:
    total_samples = len(results)
    valid_results = _valid_prediction_rows(results, labels)
    valid_samples = len(valid_results)

    if valid_samples == 0:
        raise ValueError("No valid predictions available for summary metrics.")

    output_quality = compute_output_quality(results, labels)
    y_true, y_pred = _classification_arrays(results, labels)

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = confusion.ravel()

    coverage = valid_samples / total_samples if total_samples > 0 else 0

    return {
        "run_id": run_id,
        "method": method,
        "model": model,
        "task_type": task_type,
        "dataset_file": dataset_file,
        "baseline_prompt_file": baseline_prompt_file,
        "second_level_prompt_file": second_level_prompt_file,
        "total_samples": total_samples,
        "valid_samples": valid_samples,
        "invalid_samples": total_samples - valid_samples,
        "coverage": coverage,
        "json_valid_rate": coverage,
        "parsing_errors": output_quality["parsing_errors"],
        "invalid_labels": output_quality["invalid_labels"],
        "parsing_rate": output_quality["parsing_rate"],
        "invalid_label_rate": output_quality["invalid_label_rate"],
        "json_success_rate": output_quality["json_success_rate"],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=labels[0], zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": confusion.tolist(),
    }


def compute_output_quality(results: List[Dict[str, Any]], task_labels: List[str]) -> Dict[str, Any]:
    total = len(results)
    parsing_errors = [r for r in results if r.get("predicted_label") == "parsing_error"]
    invalid_labels = [r for r in results if (r.get("predicted_label") not in task_labels) and (r.get("predicted_label") != "parsing_error")]

    parsing_rate = len(parsing_errors) / total if total else 0
    invalid_label_rate = len(invalid_labels) / total if total else 0
    json_success_rate = 1 - parsing_rate - invalid_label_rate

    return {
        "total": total,
        "parsing_errors": len(parsing_errors),
        "invalid_labels": len(invalid_labels),
        "parsing_rate": parsing_rate,
        "invalid_label_rate": invalid_label_rate,
        "json_success_rate": json_success_rate,
    }


def compute_second_level_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = results
    n_total = len(df)

    first_correct = [r.get("first_level_label") == r.get("true_label") for r in df]
    final_correct = [r.get("predicted_label") == r.get("true_label") for r in df]

    corrected = sum((not f) and fin for f, fin in zip(first_correct, final_correct))
    degraded = sum(f and (not fin) for f, fin in zip(first_correct, final_correct))
    unchanged_correct = sum(f and fin for f, fin in zip(first_correct, final_correct))
    unchanged_wrong = sum((not f) and (not fin) for f, fin in zip(first_correct, final_correct))

    n_first_wrong = sum((not f) for f in first_correct)
    n_first_correct = sum(first_correct)

    correction_rate = corrected / n_first_wrong if n_first_wrong > 0 else 0
    degradation_rate = degraded / n_first_correct if n_first_correct > 0 else 0

    first_accuracy = sum(first_correct) / n_total if n_total else 0
    final_accuracy = sum(final_correct) / n_total if n_total else 0

    valid_second_level = [r for r in df if r.get("second_level_verdict") in {"correct", "not_correct"}]
    second_level_coverage = len(valid_second_level) / n_total if n_total else 0

    if not valid_second_level:
        override_rate = 0
        agreement_rate = 0
    else:
        override_rate = sum(1 for r in valid_second_level if r.get("second_level_verdict") == "not_correct") / len(valid_second_level)
        agreement_rate = sum(1 for r in valid_second_level if r.get("second_level_verdict") == "correct") / len(valid_second_level)

    net_gain_count = corrected - degraded

    return {
        "total_samples": n_total,
        "first_level_correct": int(n_first_correct),
        "first_level_wrong": int(n_first_wrong),
        "corrected_count": int(corrected),
        "degraded_count": int(degraded),
        "unchanged_correct_count": int(unchanged_correct),
        "unchanged_wrong_count": int(unchanged_wrong),
        "correction_rate": correction_rate,
        "degradation_rate": degradation_rate,
        "first_level_accuracy": first_accuracy,
        "final_accuracy": final_accuracy,
        "accuracy_delta": final_accuracy - first_accuracy,
        "override_rate": override_rate,
        "agreement_rate": agreement_rate,
        "second_level_coverage": second_level_coverage,
        "net_gain_count": int(net_gain_count),
    }


def compute_difference_metrics(
    baseline_metrics: Dict[str, Any],
    improved_metrics: Dict[str, Any],
) -> Dict[str, float]:
    """Compute metric differences as improved - baseline.

    Positive values indicate improvement, zero means no change, and negative
    values indicate degradation.
    """
    metric_names = ("accuracy", "precision", "recall", "f1")

    differences = {}
    for name in metric_names:
        if name in baseline_metrics and name in improved_metrics:
            differences[f"{name}_difference"] = float(
                improved_metrics[name] - baseline_metrics[name]
            )

    return differences


def compute_dynamic_prompt_metrics(
    baseline_results: List[Dict[str, Any]],
    dynamic_results: List[Dict[str, Any]],
    labels: List[str],
) -> Dict[str, Any]:
    """Compute metrics comparing dynamic prompts against baseline.

    Returns:
        - difference metrics (dynamic - baseline) for accuracy, precision, recall, f1
        - agreement rate: fraction of samples where both methods agree
        - improvement rate: fraction of samples where dynamic is correct and baseline wrong
        - degradation rate: fraction where baseline is correct and dynamic wrong
    """
    if len(baseline_results) != len(dynamic_results):
        raise ValueError("Both result lists must have the same length and be aligned.")

    baseline_metrics = compute_classification_metrics(baseline_results, labels)
    dynamic_metrics = compute_classification_metrics(dynamic_results, labels)

    diffs = compute_difference_metrics(baseline_metrics, dynamic_metrics)

    # Agreement, improvement, degradation per sample
    n = len(baseline_results)
    agree = 0
    improve = 0
    degrade = 0
    for br, dr in zip(baseline_results, dynamic_results):
        b_label = br.get("predicted_label")
        d_label = dr.get("predicted_label")
        true = br.get("true_label")
        if b_label not in labels or d_label not in labels:
            continue
        b_correct = (b_label == true)
        d_correct = (d_label == true)
        if b_correct == d_correct:
            agree += 1
        elif d_correct and not b_correct:
            improve += 1
        elif b_correct and not d_correct:
            degrade += 1

    total_valid = n  # or count of valid pairs, but we use n for simplicity
    return {
        **diffs,
        "agreement_rate": agree / total_valid if total_valid else 0,
        "improvement_rate": improve / total_valid if total_valid else 0,
        "degradation_rate": degrade / total_valid if total_valid else 0,
        "agreement_count": agree,
        "improvement_count": improve,
        "degradation_count": degrade,
    }


def compute_flash_metric(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute a simple 'flash' metric: average length of model_response.

    This can be used as a proxy for response speed / verbosity.
    """
    lengths = [len(r.get("model_response", "")) for r in results]
    word_counts = [len(r.get("model_response", "").split()) for r in results]
    return {
        "avg_response_length_chars": float(np.mean(lengths)) if lengths else 0,
        "avg_response_length_words": float(np.mean(word_counts)) if word_counts else 0,
        "min_response_length": float(np.min(lengths)) if lengths else 0,
        "max_response_length": float(np.max(lengths)) if lengths else 0,
    }


def compute_confusion_matrices_by_group(
    results: List[Dict[str, Any]], labels: List[str], group_key: str
) -> Dict[str, Any]:
    groups = {}
    for r in results:
        k = r.get(group_key, "unknown")
        groups.setdefault(k, []).append(r)

    out = {}
    for k, group in groups.items():
        y_true = [x["true_label"] for x in group if x.get("predicted_label") in labels]
        y_pred = [x["predicted_label"] for x in group if x.get("predicted_label") in labels]
        if not y_true:
            out[k] = {"error": "no valid preds"}
            continue
        out[k] = {
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "counts": len(group),
        }

    return out


def sample_error_cases(results: List[Dict[str, Any]], n: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
    errors = [
        r for r in results
        if r.get("predicted_label") == "parsing_error"
        or (r.get("predicted_label") not in {r.get("true_label"), "parsing_error"})
    ]
    random.Random(seed).shuffle(errors)
    return errors[:n]


def stratified_metrics_by_length(
    results: List[Dict[str, Any]], labels: List[str], bins: Tuple[int, int] = (50, 150)
) -> Dict[str, Any]:
    short_max, medium_max = bins
    buckets = {"short": [], "medium": [], "long": []}
    for r in results:
        lr = len(r.get("model_response", ""))
        if lr <= short_max:
            buckets["short"].append(r)
        elif lr <= medium_max:
            buckets["medium"].append(r)
        else:
            buckets["long"].append(r)

    out = {}
    for name, group in buckets.items():
        out[name] = compute_classification_metrics(group, labels) if group else {"error": "no samples"}

    return out


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def mcnemar_test(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], positive_labels: List[str]) -> Dict[str, Any]:
    n01 = 0
    n10 = 0
    for ra, rb in zip(results_a, results_b):
        ya = ra.get("predicted_label")
        yb = rb.get("predicted_label")
        true = ra.get("true_label")
        if ya not in positive_labels or yb not in positive_labels:
            continue
        a_corr = (ya == true)
        b_corr = (yb == true)
        if a_corr and (not b_corr):
            n10 += 1
        elif (not a_corr) and b_corr:
            n01 += 1

    n = n01 + n10
    if n == 0:
        return {"n01": n01, "n10": n10, "stat": None, "p_value": None}

    stat = (abs(n01 - n10) - 1) ** 2 / n
    p = chi2.sf(stat, df=1)
    return {"n01": n01, "n10": n10, "stat": float(stat), "p_value": float(p)}


def compare_methods(
    results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], labels: List[str]
) -> Dict[str, Any]:
    if len(results_a) != len(results_b):
        raise ValueError("results must be same length and aligned")

    metrics_a = compute_classification_metrics(results_a, labels)
    metrics_b = compute_classification_metrics(results_b, labels)

    boot = bootstrap_paired_diff(results_a, results_b, labels, metric="accuracy", n_bootstrap=1000)
    boot_f1 = bootstrap_paired_diff(results_a, results_b, labels, metric="f1", n_bootstrap=1000)
    m_test = mcnemar_test(results_a, results_b, labels)

    return {
        "method_a": metrics_a,
        "method_b": metrics_b,
        "accuracy_diff_bootstrap": boot,
        "f1_diff_bootstrap": boot_f1,
        "mcnemar": m_test,
    }


def bootstrap_paired_diff(
    baseline_results: List[Dict[str, Any]],
    improved_results: List[Dict[str, Any]],
    labels: List[str],
    metric: str = "accuracy",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    if len(baseline_results) != len(improved_results):
        raise ValueError("baseline_results and improved_results must have the same length")

    if metric not in {"accuracy", "f1"}:
        raise ValueError(f"Unsupported metric: {metric}")

    rng = random.Random(seed)
    n = len(baseline_results)

    if n == 0:
        return {
            "metric": metric,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n_bootstrap": n_bootstrap,
            "paired_valid_samples": 0,
            "paired_valid_coverage": 0.0,
        }

    y_true = [r.get("true_label") for r in baseline_results]

    valid_mask = [
        (
            baseline_results[i].get("predicted_label") in labels
            and improved_results[i].get("predicted_label") in labels
        )
        for i in range(n)
    ]
    paired_valid_samples = sum(valid_mask)

    diffs = []

    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]

        y_true_bs = [y_true[i] for i in idx]
        baseline_pred = [baseline_results[i].get("predicted_label") for i in idx]
        improved_pred = [improved_results[i].get("predicted_label") for i in idx]

        if metric == "accuracy":
            baseline_metric = accuracy_score(y_true_bs, baseline_pred)
            improved_metric = accuracy_score(y_true_bs, improved_pred)
        else:  # f1
            valid_idx = [i for i in idx if valid_mask[i]]
            if not valid_idx:
                diffs.append(0.0)
                continue
            y_true_valid = [y_true[i] for i in valid_idx]
            baseline_valid = [baseline_results[i].get("predicted_label") for i in valid_idx]
            improved_valid = [improved_results[i].get("predicted_label") for i in valid_idx]
            baseline_metric = f1_score(y_true_valid, baseline_valid, zero_division=0, pos_label=labels[0])
            improved_metric = f1_score(y_true_valid, improved_valid, zero_division=0, pos_label=labels[0])

        diffs.append(improved_metric - baseline_metric)

    diffs_arr = np.array(diffs)

    return {
        "metric": metric,
        "mean_diff": float(diffs_arr.mean()),
        "ci_lower": float(np.percentile(diffs_arr, 2.5)),
        "ci_upper": float(np.percentile(diffs_arr, 97.5)),
        "n_bootstrap": n_bootstrap,
        "paired_valid_samples": paired_valid_samples,
        "paired_valid_coverage": paired_valid_samples / n,
    }