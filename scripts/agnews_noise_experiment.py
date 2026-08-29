"""Construct and evaluate a controlled noisy-label AG News dataset.

This creates a derived dataset called AGNews-CLNoise:
- 10,000 stratified AG News training examples
- Four balanced classes
- Exactly 20% deliberately corrupted labels
- Fixed cyclic class-conditional transitions
- Original AG News labels treated as the clean reference

Raw news text is not committed. The manifest records source indices,
labels and SHA-256 text hashes.
"""

from pathlib import Path
import csv
import hashlib
import json
import random

import cleanlab
import datasets
import numpy as np
import sklearn
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_SEED = 42
SUBSET_SIZE = 10000
NOISE_RATE = 0.20
CROSS_VALIDATION_FOLDS = 4

DATASET_ID = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
DATASET_NAME = "AGNews-CLNoise"

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# Artificial cyclic corruption chosen for reproducibility, not because
# these transitions are claimed to model realistic annotation mistakes.
LABEL_TRANSITION = {
    0: 1,  # World -> Sports
    1: 2,  # Sports -> Business
    2: 3,  # Business -> Sci/Tech
    3: 0,  # Sci/Tech -> World
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw" / "huggingface"
PROCESSED_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "processed"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

MANIFEST_FILE = (
    PROCESSED_DATA_DIRECTORY / "agnews_clnoise_manifest.csv"
)
SUMMARY_FILE = RESULTS_DIRECTORY / "agnews_noise_summary.json"
PREDICTIONS_FILE = (
    RESULTS_DIRECTORY / "agnews_noise_predictions.csv"
)


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


def calculate_metrics(flagged, actual_errors):
    true_positives = int(np.sum(flagged & actual_errors))
    false_positives = int(np.sum(flagged & ~actual_errors))
    false_negatives = int(np.sum(~flagged & actual_errors))

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

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

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
        raise ValueError(
            f"Unexpected dataset columns: {dataset.column_names}"
        )

    dataset_class_names = dataset.features["label"].names

    if list(dataset_class_names) != CLASS_NAMES:
        raise ValueError(
            f"Unexpected class names: {dataset_class_names}"
        )

    all_labels = np.asarray(dataset["label"], dtype=np.int64)
    all_indices = np.arange(len(dataset))

    subset_indices, _ = train_test_split(
        all_indices,
        train_size=SUBSET_SIZE,
        random_state=RANDOM_SEED,
        stratify=all_labels,
    )
    subset_indices = np.sort(subset_indices)

    subset = dataset.select(subset_indices.tolist())
    texts = list(subset["text"])
    clean_labels = np.asarray(subset["label"], dtype=np.int64)

    clean_class_counts = np.bincount(
        clean_labels,
        minlength=len(CLASS_NAMES),
    )

    expected_per_class = SUBSET_SIZE // len(CLASS_NAMES)

    if not np.array_equal(
        clean_class_counts,
        np.full(len(CLASS_NAMES), expected_per_class),
    ):
        raise ValueError(
            f"Unexpected subset class counts: {clean_class_counts}"
        )

    rng = np.random.default_rng(RANDOM_SEED)
    noisy_labels = clean_labels.copy()
    injected_error_mask = np.zeros(SUBSET_SIZE, dtype=bool)

    errors_per_class = int(expected_per_class * NOISE_RATE)

    for source_class, destination_class in LABEL_TRANSITION.items():
        class_positions = np.flatnonzero(
            clean_labels == source_class
        )
        selected_positions = rng.choice(
            class_positions,
            size=errors_per_class,
            replace=False,
        )

        noisy_labels[selected_positions] = destination_class
        injected_error_mask[selected_positions] = True

    expected_errors = int(SUBSET_SIZE * NOISE_RATE)

    if int(injected_error_mask.sum()) != expected_errors:
        raise RuntimeError(
            "The number of injected errors is not exactly 20%."
        )

    if not np.array_equal(
        injected_error_mask,
        noisy_labels != clean_labels,
    ):
        raise RuntimeError(
            "Injected-error mask does not match changed labels."
        )

    print("--- AGNews-CLNoise construction ---")
    print("Source dataset:", DATASET_ID)
    print("Source revision:", DATASET_REVISION)
    print("Source training examples:", len(dataset))
    print("Constructed subset:", SUBSET_SIZE)
    print("Clean class counts:", clean_class_counts)
    print("Injected errors:", int(injected_error_mask.sum()))
    print(f"Injected noise rate: {injected_error_mask.mean():.4f}")
    print("Errors per class:", errors_per_class)

    classifier = make_pipeline(
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            max_features=30000,
            sublinear_tf=True,
        ),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
    )

    cross_validation = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    print()
    print(
        "Generating four-fold out-of-sample "
        "TF-IDF logistic-regression probabilities..."
    )

    predicted_probabilities = cross_val_predict(
        classifier,
        texts,
        noisy_labels,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=2,
    )

    predicted_labels = np.argmax(
        predicted_probabilities,
        axis=1,
    )

    issue_output = find_label_issues(
        labels=noisy_labels,
        pred_probs=predicted_probabilities,
        n_jobs=2,
    )

    flagged_issue_mask = convert_to_issue_mask(
        issue_output,
        SUBSET_SIZE,
    )

    label_quality_scores = get_label_quality_scores(
        labels=noisy_labels,
        pred_probs=predicted_probabilities,
    )

    metrics = calculate_metrics(
        flagged_issue_mask,
        injected_error_mask,
    )

    accuracy_against_noisy = accuracy_score(
        noisy_labels,
        predicted_labels,
    )
    accuracy_against_clean = accuracy_score(
        clean_labels,
        predicted_labels,
    )

    transition_description = {
        CLASS_NAMES[source]: CLASS_NAMES[destination]
        for source, destination in LABEL_TRANSITION.items()
    }

    summary = {
        "constructed_dataset_name": DATASET_NAME,
        "description": (
            "Controlled class-conditional label-noise dataset "
            "derived from AG News"
        ),
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "source_fingerprint": dataset._fingerprint,
        "source_training_examples": len(dataset),
        "subset_size": SUBSET_SIZE,
        "random_seed": RANDOM_SEED,
        "class_names": CLASS_NAMES,
        "clean_class_counts": clean_class_counts.tolist(),
        "noise_type": (
            "fixed cyclic class-conditional asymmetric noise"
        ),
        "label_transitions": transition_description,
        "noise_rate": NOISE_RATE,
        "errors_per_class": errors_per_class,
        "injected_label_errors": int(
            injected_error_mask.sum()
        ),
        "cross_validation_folds": CROSS_VALIDATION_FOLDS,
        "suspected_label_issues": int(
            flagged_issue_mask.sum()
        ),
        **metrics,
        "accuracy_against_noisy_labels": float(
            accuracy_against_noisy
        ),
        "accuracy_against_clean_labels": float(
            accuracy_against_clean
        ),
        "cleanlab_version": cleanlab.__version__,
        "datasets_version": datasets.__version__,
        "scikit_learn_version": sklearn.__version__,
        "raw_text_committed": False,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file,
            lineterminator="\n",
        )
        writer.writerow(
            [
                "source_index",
                "text_sha256",
                "clean_label",
                "clean_class",
                "noisy_label",
                "noisy_class",
                "injected_label_error",
            ]
        )

        for position, source_index in enumerate(subset_indices):
            writer.writerow(
                [
                    int(source_index),
                    text_hash(texts[position]),
                    int(clean_labels[position]),
                    CLASS_NAMES[clean_labels[position]],
                    int(noisy_labels[position]),
                    CLASS_NAMES[noisy_labels[position]],
                    bool(injected_error_mask[position]),
                ]
            )

    with PREDICTIONS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(
            output_file,
            lineterminator="\n",
        )
        writer.writerow(
            [
                "source_index",
                "clean_label",
                "noisy_label",
                "predicted_label",
                "injected_label_error",
                "cleanlab_flagged",
                "label_quality_score",
            ]
        )

        for position, source_index in enumerate(subset_indices):
            writer.writerow(
                [
                    int(source_index),
                    int(clean_labels[position]),
                    int(noisy_labels[position]),
                    int(predicted_labels[position]),
                    bool(injected_error_mask[position]),
                    bool(flagged_issue_mask[position]),
                    float(label_quality_scores[position]),
                ]
            )

    print()
    print("--- AGNews-CLNoise results ---")
    print(
        "Actual injected errors:",
        int(injected_error_mask.sum()),
    )
    print(
        "Cleanlab suspected issues:",
        int(flagged_issue_mask.sum()),
    )
    print("True positives:", metrics["true_positives"])
    print("False positives:", metrics["false_positives"])
    print("False negatives:", metrics["false_negatives"])
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")
    print(
        "Accuracy against noisy labels:",
        f"{accuracy_against_noisy:.4f}",
    )
    print(
        "Accuracy against clean labels:",
        f"{accuracy_against_clean:.4f}",
    )
    print()
    print("Manifest saved to:", MANIFEST_FILE)
    print("Summary saved to:", SUMMARY_FILE)
    print("Predictions saved to:", PREDICTIONS_FILE)


if __name__ == "__main__":
    main()