"""Controlled sensitivity study for the derived AGNews-CLNoise dataset.

Runs 4 noise rates x 2 corruption mechanisms x 3 corruption seeds.
The balanced 10,000-article subset and cross-validation folds are fixed.
The 20%-cyclic-seed-42 configuration reproduces the original pilot.
"""

from pathlib import Path
import csv
import hashlib
import json
import random

import cleanlab
import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline


SUBSET_SEED = 42
FOLD_SEED = 42
SUBSET_SIZE = 10_000
CROSS_VALIDATION_FOLDS = 4
NOISE_RATES = (0.10, 0.20, 0.30, 0.40)
NOISE_TYPES = ("cyclic", "symmetric")
CORRUPTION_SEEDS = (42, 43, 44)

DATASET_ID = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_TRANSITION = {0: 1, 1: 2, 2: 3, 3: 0}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "huggingface"
PRIVATE_REVIEW_FILE = (
    REPOSITORY_ROOT / "data" / "raw" / "agnews_manual_annotation_private.csv"
)
PROCESSED_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

RUNS_FILE = RESULTS_DIRECTORY / "agnews_noise_sweep_runs.csv"
AGGREGATE_FILE = RESULTS_DIRECTORY / "agnews_noise_sweep_aggregate.csv"
SUMMARY_FILE = RESULTS_DIRECTORY / "agnews_noise_sweep_summary.json"
MANIFEST_FILE = PROCESSED_DATA_DIRECTORY / "agnews_noise_sweep_manifest.csv"
REVIEW_KEY_FILE = PROCESSED_DATA_DIRECTORY / "agnews_manual_annotation_key.csv"
CHART_FILE = RESULTS_DIRECTORY / "agnews_noise_sweep_f1.png"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def convert_to_issue_mask(issue_output, number_of_examples):
    issue_output = np.asarray(issue_output)
    if (
        issue_output.dtype == bool
        and issue_output.shape == (number_of_examples,)
    ):
        return issue_output
    issue_mask = np.zeros(number_of_examples, dtype=bool)
    issue_mask[issue_output.astype(int)] = True
    return issue_mask


def calculate_metrics(flagged, actual_errors, issue_scores=None):
    flagged = np.asarray(flagged, dtype=bool)
    actual_errors = np.asarray(actual_errors, dtype=bool)
    true_positives = int(np.sum(flagged & actual_errors))
    false_positives = int(np.sum(flagged & ~actual_errors))
    false_negatives = int(np.sum(~flagged & actual_errors))
    true_negatives = int(np.sum(~flagged & ~actual_errors))
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    result = {
        "flagged": int(flagged.sum()),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "mcc": float(matthews_corrcoef(actual_errors, flagged)),
    }
    if issue_scores is not None:
        result["average_precision"] = float(
            average_precision_score(actual_errors, issue_scores)
        )
    return result


def corrupt_labels(clean_labels, noise_rate, noise_type, seed):
    """Corrupt exactly the same number of examples per source class."""
    selection_rng = np.random.default_rng(seed)
    destination_rng = np.random.default_rng(seed + 1_000_003)
    noisy_labels = clean_labels.copy()
    injected_error_mask = np.zeros(len(clean_labels), dtype=bool)
    errors_per_class = int(
        (len(clean_labels) // len(CLASS_NAMES)) * noise_rate
    )

    for source_class in range(len(CLASS_NAMES)):
        class_positions = np.flatnonzero(clean_labels == source_class)
        selected_positions = selection_rng.choice(
            class_positions,
            size=errors_per_class,
            replace=False,
        )
        if noise_type == "cyclic":
            destinations = np.full(
                errors_per_class,
                LABEL_TRANSITION[source_class],
                dtype=np.int64,
            )
        elif noise_type == "symmetric":
            alternatives = np.asarray(
                [
                    label
                    for label in range(len(CLASS_NAMES))
                    if label != source_class
                ],
                dtype=np.int64,
            )
            destinations = destination_rng.choice(
                alternatives,
                size=errors_per_class,
                replace=True,
            )
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        noisy_labels[selected_positions] = destinations
        injected_error_mask[selected_positions] = True

    expected_errors = int(len(clean_labels) * noise_rate)
    if int(injected_error_mask.sum()) != expected_errors:
        raise RuntimeError("Unexpected number of injected errors.")
    if not np.array_equal(
        injected_error_mask,
        noisy_labels != clean_labels,
    ):
        raise RuntimeError("Error mask does not match changed labels.")
    return noisy_labels, injected_error_mask


def make_classifier(seed):
    return make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=30_000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        ),
    )


def add_prefixed_metrics(row, prefix, metrics, prevalence):
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value
    row[f"{prefix}_lift"] = (
        metrics["precision"] / prevalence if prevalence else 0.0
    )


def build_chart(aggregate):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colours = {"cleanlab": "#2166ac", "argmax": "#b2182b"}
    labels = {
        "cleanlab": "Cleanlab",
        "argmax": "Argmax disagreement",
    }
    for axis, noise_type in zip(axes, NOISE_TYPES):
        subset = aggregate[aggregate["noise_type"] == noise_type]
        for method in ("cleanlab", "argmax"):
            axis.errorbar(
                subset["noise_rate"] * 100,
                subset[f"{method}_f1_score_mean"],
                yerr=subset[f"{method}_f1_score_std"],
                marker="o",
                linewidth=2,
                capsize=4,
                color=colours[method],
                label=labels[method],
            )
        axis.set_title(f"{noise_type.title()} corruption")
        axis.set_xlabel("Injected label-noise rate (%)")
        axis.set_xticks([10, 20, 30, 40])
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Error-detection F1")
    axes[1].legend(loc="best")
    fig.suptitle("AGNews-CLNoise sensitivity study (mean ± SD, 3 seeds)")
    fig.tight_layout()
    fig.savefig(CHART_FILE, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    random.seed(SUBSET_SEED)
    np.random.seed(SUBSET_SEED)
    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(RAW_DATA_DIRECTORY),
    )
    if dataset.column_names != ["text", "label"]:
        raise ValueError(f"Unexpected columns: {dataset.column_names}")
    if list(dataset.features["label"].names) != CLASS_NAMES:
        raise ValueError("Unexpected AG News class names.")

    all_labels = np.asarray(dataset["label"], dtype=np.int64)
    all_indices = np.arange(len(dataset))
    subset_indices, _ = train_test_split(
        all_indices,
        train_size=SUBSET_SIZE,
        random_state=SUBSET_SEED,
        stratify=all_labels,
    )
    subset_indices = np.sort(subset_indices)
    subset = dataset.select(subset_indices.tolist())
    texts = list(subset["text"])
    clean_labels = np.asarray(subset["label"], dtype=np.int64)
    text_hashes = [text_hash(text) for text in texts]

    clean_counts = np.bincount(clean_labels, minlength=len(CLASS_NAMES))
    if not np.array_equal(clean_counts, np.full(4, 2500)):
        raise RuntimeError(f"Unexpected class counts: {clean_counts}")

    # Build folds from the verified 20%-cyclic-seed-42 baseline and reuse
    # them for every run. This both reproduces the pilot and fixes folds.
    reference_noisy_labels, _ = corrupt_labels(
        clean_labels, 0.20, "cyclic", 42
    )
    splitter = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=FOLD_SEED,
    )
    fixed_folds = list(splitter.split(texts, reference_noisy_labels))

    run_rows = []
    manifest_rows = []
    baseline_review_data = None
    total_runs = len(NOISE_RATES) * len(NOISE_TYPES) * len(CORRUPTION_SEEDS)
    run_number = 0

    print("--- AGNews-CLNoise controlled sensitivity study ---")
    print("Source revision:", DATASET_REVISION)
    print("Fixed subset:", SUBSET_SIZE, "articles")
    print("Clean class counts:", clean_counts.tolist())
    print("Runs:", total_runs)

    for noise_type in NOISE_TYPES:
        for noise_rate in NOISE_RATES:
            for corruption_seed in CORRUPTION_SEEDS:
                run_number += 1
                run_id = (
                    f"{noise_type}_r{int(noise_rate * 100):02d}"
                    f"_s{corruption_seed}"
                )
                print(f"[{run_number:02d}/{total_runs}] {run_id}")

                noisy_labels, actual_errors = corrupt_labels(
                    clean_labels,
                    noise_rate,
                    noise_type,
                    corruption_seed,
                )
                probabilities = cross_val_predict(
                    make_classifier(SUBSET_SEED),
                    texts,
                    noisy_labels,
                    cv=fixed_folds,
                    method="predict_proba",
                    n_jobs=2,
                )
                predicted_labels = np.argmax(probabilities, axis=1)

                issue_output = find_label_issues(
                    labels=noisy_labels,
                    pred_probs=probabilities,
                    n_jobs=2,
                )
                cleanlab_mask = convert_to_issue_mask(
                    issue_output, SUBSET_SIZE
                )
                argmax_mask = predicted_labels != noisy_labels
                quality_scores = get_label_quality_scores(
                    labels=noisy_labels,
                    pred_probs=probabilities,
                )
                issue_scores = 1.0 - quality_scores

                cleanlab_metrics = calculate_metrics(
                    cleanlab_mask, actual_errors, issue_scores
                )
                argmax_metrics = calculate_metrics(
                    argmax_mask, actual_errors
                )
                prevalence = float(actual_errors.mean())
                row = {
                    "run_id": run_id,
                    "noise_type": noise_type,
                    "noise_rate": noise_rate,
                    "corruption_seed": corruption_seed,
                    "subset_size": SUBSET_SIZE,
                    "injected_errors": int(actual_errors.sum()),
                    "accuracy_noisy": float(
                        accuracy_score(noisy_labels, predicted_labels)
                    ),
                    "accuracy_clean": float(
                        accuracy_score(clean_labels, predicted_labels)
                    ),
                }
                add_prefixed_metrics(
                    row, "cleanlab", cleanlab_metrics, prevalence
                )
                add_prefixed_metrics(
                    row, "argmax", argmax_metrics, prevalence
                )
                run_rows.append(row)

                for position in np.flatnonzero(actual_errors):
                    manifest_rows.append(
                        {
                            "run_id": run_id,
                            "source_index": int(subset_indices[position]),
                            "text_sha256": text_hashes[position],
                            "clean_label": int(clean_labels[position]),
                            "clean_class": CLASS_NAMES[clean_labels[position]],
                            "noisy_label": int(noisy_labels[position]),
                            "noisy_class": CLASS_NAMES[noisy_labels[position]],
                        }
                    )

                if (
                    noise_type == "cyclic"
                    and noise_rate == 0.20
                    and corruption_seed == 42
                ):
                    baseline_review_data = {
                        "cleanlab_mask": cleanlab_mask.copy(),
                        "actual_errors": actual_errors.copy(),
                        "predicted_labels": predicted_labels.copy(),
                    }
                    expected = {
                        "flagged": 2251,
                        "true_positives": 1701,
                        "false_positives": 550,
                        "false_negatives": 299,
                    }
                    for key, expected_value in expected.items():
                        if cleanlab_metrics[key] != expected_value:
                            raise RuntimeError(
                                "Baseline reproduction failed: "
                                f"{key}={cleanlab_metrics[key]}, "
                                f"expected {expected_value}."
                            )

    runs = pd.DataFrame(run_rows)
    runs.to_csv(RUNS_FILE, index=False, lineterminator="\n")
    pd.DataFrame(manifest_rows).to_csv(
        MANIFEST_FILE, index=False, lineterminator="\n"
    )

    metric_columns = [
        "cleanlab_precision",
        "cleanlab_recall",
        "cleanlab_f1_score",
        "cleanlab_mcc",
        "cleanlab_average_precision",
        "cleanlab_lift",
        "argmax_precision",
        "argmax_recall",
        "argmax_f1_score",
        "argmax_mcc",
        "argmax_lift",
    ]
    aggregate = (
        runs.groupby(["noise_type", "noise_rate"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = [
        column[0]
        if not column[1]
        else f"{column[0]}_{column[1]}"
        for column in aggregate.columns
    ]
    aggregate.to_csv(AGGREGATE_FILE, index=False, lineterminator="\n")
    build_chart(aggregate)

    if baseline_review_data is None:
        raise RuntimeError("Baseline review data was not retained.")
    false_positive_positions = np.flatnonzero(
        baseline_review_data["cleanlab_mask"]
        & ~baseline_review_data["actual_errors"]
    )
    review_rng = np.random.default_rng(2026)
    review_positions = []
    target_per_class = [8, 8, 7, 7]
    for class_label, target in enumerate(target_per_class):
        candidates = false_positive_positions[
            clean_labels[false_positive_positions] == class_label
        ]
        review_positions.extend(
            review_rng.choice(candidates, size=target, replace=False).tolist()
        )
    review_positions = np.asarray(review_positions, dtype=int)
    review_rng.shuffle(review_positions)

    with PRIVATE_REVIEW_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "review_id",
                "source_index",
                "article_text",
                "original_agnews_label",
                "human_decision",
                "reason",
            ]
        )
        for review_id, position in enumerate(review_positions, start=1):
            writer.writerow(
                [
                    review_id,
                    int(subset_indices[position]),
                    texts[position],
                    CLASS_NAMES[clean_labels[position]],
                    "",
                    "",
                ]
            )

    with REVIEW_KEY_FILE.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "review_id",
                "source_index",
                "text_sha256",
                "original_agnews_label",
            ]
        )
        for review_id, position in enumerate(review_positions, start=1):
            writer.writerow(
                [
                    review_id,
                    int(subset_indices[position]),
                    text_hashes[position],
                    CLASS_NAMES[clean_labels[position]],
                ]
            )

    summary = {
        "experiment": "AGNews-CLNoise controlled sensitivity study",
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "subset_size": SUBSET_SIZE,
        "subset_seed": SUBSET_SEED,
        "fold_seed": FOLD_SEED,
        "folds_fixed_across_runs": True,
        "noise_rates": list(NOISE_RATES),
        "noise_types": list(NOISE_TYPES),
        "corruption_seeds": list(CORRUPTION_SEEDS),
        "total_runs": total_runs,
        "baseline": "argmax prediction differs from supplied noisy label",
        "manual_review_cases": len(review_positions),
        "cleanlab_version": cleanlab.__version__,
        "datasets_version": datasets.__version__,
        "scikit_learn_version": sklearn.__version__,
        "raw_text_committed": False,
        "outputs": {
            "runs": str(RUNS_FILE.relative_to(REPOSITORY_ROOT)),
            "aggregate": str(AGGREGATE_FILE.relative_to(REPOSITORY_ROOT)),
            "manifest": str(MANIFEST_FILE.relative_to(REPOSITORY_ROOT)),
            "chart": str(CHART_FILE.relative_to(REPOSITORY_ROOT)),
            "manual_review_key": str(
                REVIEW_KEY_FILE.relative_to(REPOSITORY_ROOT)
            ),
        },
    }
    with SUMMARY_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print()
    print("--- Aggregate F1 results ---")
    print(
        aggregate[
            [
                "noise_type",
                "noise_rate",
                "cleanlab_f1_score_mean",
                "cleanlab_f1_score_std",
                "argmax_f1_score_mean",
                "argmax_f1_score_std",
            ]
        ].to_string(index=False)
    )
    print()
    print("Baseline 20%-cyclic-seed-42: EXACT MATCH")
    print("Runs saved to:", RUNS_FILE)
    print("Aggregates saved to:", AGGREGATE_FILE)
    print("Manifest saved to:", MANIFEST_FILE)
    print("Chart saved to:", CHART_FILE)
    print("Private annotation sheet:", PRIVATE_REVIEW_FILE)
    print("Public annotation key:", REVIEW_KEY_FILE)


if __name__ == "__main__":
    main()
