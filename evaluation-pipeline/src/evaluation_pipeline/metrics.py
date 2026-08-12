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
from scipy.stats import binomtest
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
    return [
        r
        for r in results
        if r.get("predicted_label") in labels
    ]


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
    # Operates on list of results (dicts) and mirrors notebook logic.
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


def compute_confusion_matrices_by_group(results: List[Dict[str, Any]], labels: List[str], group_key: str) -> Dict[str, Any]:
    # group_key can be 'method' or any key present in result dict (e.g., provided prompt_version)
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
    errors = [r for r in results if r.get("predicted_label") == "parsing_error" or (r.get("predicted_label") not in {r.get("true_label"), "parsing_error"})]
    random.Random(seed).shuffle(errors)
    return errors[:n]


def stratified_metrics_by_length(results: List[Dict[str, Any]], labels: List[str], bins: Tuple[int, int] = (50, 150)) -> Dict[str, Any]:
    # bins: (short_max, medium_max). short: <=short_max, medium: <=medium_max, long: > medium_max
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


def mcnemar_test(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute McNemar's test for paired binary outcomes.

    Expects `results_a` and `results_b` to be lists of dicts aligned per-sample
    and containing `true_label` and `predicted_label`.

    Returns contingency counts and two-sided exact p-value.
    """
    if len(results_a) != len(results_b):
        raise ValueError("results lists must have the same length")

    b = 0  # a correct, b wrong
    c = 0  # a wrong, b correct

    for ra, rb in zip(results_a, results_b):
        true = ra.get("true_label")
        pa = ra.get("predicted_label")
        pb = rb.get("predicted_label")

        a_correct = (pa == true)
        b_correct = (pb == true)

        if a_correct and (not b_correct):
            b += 1
        elif (not a_correct) and b_correct:
            c += 1

    n = b + c
    if n == 0:
        pvalue = 1.0
    else:
        # exact binomial test on the smaller of b,c with p=0.5 (two-sided)
        k = min(b, c)
        pvalue = binomtest(k, n, p=0.5, alternative='two-sided').pvalue

    return {"b": b, "c": c, "n": n, "p_value": float(pvalue)}


def bootstrap_paired_diff(
    results_a: List[Dict[str, Any]],
    results_b: List[Dict[str, Any]],
    labels: List[str],
    metric: str = "accuracy",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute paired bootstrap CI for difference (a - b) of a given metric.

    Supported metrics: 'accuracy', 'f1'. Accuracy uses all aligned samples,
    while F1 is computed on samples where both predictions are valid labels.
    """
    rng = random.Random(seed)

    if len(results_a) != len(results_b):
        raise ValueError("results lists must have the same length")

    n = len(results_a)
    y_true = [r.get("true_label") for r in results_a]
    valid_mask = [
        (ra.get("predicted_label") in labels) and (rb.get("predicted_label") in labels)
        for ra, rb in zip(results_a, results_b)
    ]
    paired_valid_samples = sum(valid_mask)

    diffs = []
    for _ in range(n_bootstrap):
        idx = [rng.randrange(n) for _ in range(n)]
        y_true_bs = [y_true[i] for i in idx]

        pa = [results_a[i].get("predicted_label") for i in idx]
        pb = [results_b[i].get("predicted_label") for i in idx]

        if metric == "accuracy":
            ma = accuracy_score(y_true_bs, pa)
            mb = accuracy_score(y_true_bs, pb)
        elif metric == "f1":
            valid_idx = [i for i in idx if valid_mask[i]]
            if not valid_idx:
                diffs.append(0.0)
                continue

            y_true_valid = [y_true[i] for i in valid_idx]
            pa_valid = [results_a[i].get("predicted_label") for i in valid_idx]
            pb_valid = [results_b[i].get("predicted_label") for i in valid_idx]

            ma = f1_score(y_true_valid, pa_valid, zero_division=0, pos_label=labels[0])
            mb = f1_score(y_true_valid, pb_valid, zero_division=0, pos_label=labels[0])
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        diffs.append(ma - mb)

    diffs_arr = np.array(diffs)
    lower = float(np.percentile(diffs_arr, 2.5))
    upper = float(np.percentile(diffs_arr, 97.5))
    mean = float(diffs_arr.mean())

    return {
        "metric": metric,
        "mean_diff": mean,
        "ci_lower": lower,
        "ci_upper": upper,
        "n_bootstrap": n_bootstrap,
        "paired_valid_samples": paired_valid_samples,
        "paired_valid_coverage": paired_valid_samples / n if n else 0,
    }



def brier_score(results: List[Dict[str, Any]], positive_label: str = "truthful", prob_key: str = "confidence") -> float | None:
    """Compute Brier score for binary label when probabilities are available.

    results: list of result dicts with keys 'true_label' and `prob_key` giving probability for positive_label.
    Returns None if probabilities not available.
    """
    probs = []
    ys = []
    for r in results:
        p = r.get(prob_key)
        if p is None:
            return None
        probs.append(float(p))
        ys.append(1 if r.get("true_label") == positive_label else 0)

    probs = np.array(probs)
    ys = np.array(ys)
    return float(np.mean((probs - ys) ** 2))


def expected_calibration_error(results: List[Dict[str, Any]], n_bins: int = 10, positive_label: str = "truthful", prob_key: str = "confidence") -> float | None:
    """Compute ECE with equal-width bins. Returns None if probs missing."""
    probs = []
    ys = []
    for r in results:
        p = r.get(prob_key)
        if p is None:
            return None
        probs.append(float(p))
        ys.append(1 if r.get("true_label") == positive_label else 0)

    probs = np.array(probs)
    ys = np.array(ys)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        if not np.any(mask):
            continue
        avg_conf = probs[mask].mean()
        avg_acc = ys[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(avg_conf - avg_acc)

    return float(ece)


def confidence_coverage_curve(results: List[Dict[str, Any]], prob_key: str = "confidence", positive_label: str = "truthful") -> List[Tuple[float, float]]:
    """Return list of (threshold, coverage) where coverage is fraction of samples with prob>=threshold that are correct.
    Requires `prob_key` present, otherwise returns empty list.
    """
    probs = []
    correct = []
    for r in results:
        p = r.get(prob_key)
        if p is None:
            return []
        probs.append(float(p))
        correct.append(1 if r.get("predicted_label") == r.get("true_label") else 0)

    probs = np.array(probs)
    correct = np.array(correct)
    thresholds = np.linspace(0.0, 1.0, 21)
    out = []
    for t in thresholds:
        mask = probs >= t
        if mask.sum() == 0:
            out.append((t, None))
            continue
        out.append((t, float(correct[mask].mean())))
    return out


def _accuracy(results: List[Dict[str, Any]], labels: List[str]) -> float:
    m = compute_classification_metrics(results, labels)
    if "error" in m:
        return 0.0
    return float(m.get("accuracy", 0.0))


def bootstrap_paired_ci(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], labels: List[str], metric_fn=_accuracy, n_bootstrap: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Paired bootstrap CI for metric difference (a - b). Assumes aligned order and equal length."""
    if len(results_a) != len(results_b):
        raise ValueError("results_a and results_b must have same length for paired bootstrap")

    rng = random.Random(seed)
    n = len(results_a)
    diffs = []
    for _ in range(n_bootstrap):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_a = [results_a[i] for i in idxs]
        sample_b = [results_b[i] for i in idxs]
        va = metric_fn(sample_a, labels)
        vb = metric_fn(sample_b, labels)
        diffs.append(va - vb)

    diffs = np.array(diffs)
    lower = float(np.percentile(diffs, 2.5))
    upper = float(np.percentile(diffs, 97.5))
    mean = float(diffs.mean())
    return {"mean_diff": mean, "ci_lower": lower, "ci_upper": upper}


def mcnemar_test(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], positive_labels: List[str]) -> Dict[str, Any]:
    """Compute McNemar's test for paired binary predictions between two result lists.
    Returns test statistic and p-value.
    Only considers samples where both predictions are in positive_labels set.
    """
    if len(results_a) != len(results_b):
        raise ValueError("results must be same length")

    b = 0  # a correct, b wrong? We'll count discordant pairs: n01 and n10
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


def compare_methods(results_a: List[Dict[str, Any]], results_b: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    """Compare two methods on the same ordered dataset (paired). Returns summary table and stats."""
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
