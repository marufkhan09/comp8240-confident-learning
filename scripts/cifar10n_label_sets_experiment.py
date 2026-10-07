"""Compare all five CIFAR-10N human-label sets on one fixed subset."""

from pathlib import Path
import hashlib
import json
import subprocess

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 42
N_FOLDS = 4
SUBSET_SIZE = 5000

LABEL_SETS = [
    ("aggre_label", "Aggregate"),
    ("worse_label", "Worst-case"),
    ("random_label1", "Annotator 1"),
    ("random_label2", "Annotator 2"),
    ("random_label3", "Annotator 3"),
]

EXPECTED_SOURCE_COMMIT = (
    "49df7d8a69e355470c77c1c2f2424916325a394b"
)

ROOT = Path(__file__).resolve().parents[1]

SOURCE_REPOSITORY = (
    ROOT / "external/cifar-10-100n"
)
LABEL_FILE = (
    SOURCE_REPOSITORY
    / "data/CIFAR-10_human.pt"
)
CACHE_FILE = (
    ROOT
    / "data/raw/cifar10n_resnet18_n5000_size96.npz"
)

RESULTS_FILE = (
    ROOT / "results/cifar10n_label_sets_results.csv"
)
PREDICTIONS_FILE = (
    ROOT / "results/cifar10n_label_sets_predictions.csv"
)
SUMMARY_FILE = (
    ROOT / "results/cifar10n_label_sets_summary.json"
)
CHART_FILE = (
    ROOT / "results/cifar10n_label_sets_metrics.png"
)
HASH_FILE = (
    ROOT / "results/cifar10n_label_sets_hashes.txt"
)


def convert_to_issue_mask(issue_output, number_of_examples):
    issue_output = np.asarray(issue_output)

    if (
        issue_output.dtype == bool
        and issue_output.shape == (number_of_examples,)
    ):
        return issue_output

    issue_mask = np.zeros(
        number_of_examples,
        dtype=bool,
    )
    issue_mask[
        issue_output.astype(int)
    ] = True

    return issue_mask


def calculate_detection_metrics(
    suspected,
    actual,
    quality_scores,
):
    true_positives = int(
        np.sum(suspected & actual)
    )
    false_positives = int(
        np.sum(suspected & ~actual)
    )
    false_negatives = int(
        np.sum(~suspected & actual)
    )
    true_negatives = int(
        np.sum(~suspected & ~actual)
    )

    precision = (
        true_positives
        / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )

    recall = (
        true_positives
        / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )

    f1_score = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    mcc = matthews_corrcoef(
        actual.astype(int),
        suspected.astype(int),
    )

    # Low label quality indicates a more likely error.
    error_scores = 1.0 - quality_scores

    average_precision = average_precision_score(
        actual.astype(int),
        error_scores,
    )

    prevalence = float(np.mean(actual))

    lift = (
        precision / prevalence
        if prevalence
        else 0.0
    )

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "matthews_correlation_coefficient":
            float(mcc),
        "average_precision": float(
            average_precision
        ),
        "error_prevalence": prevalence,
        "precision_lift_over_prevalence":
            float(lift),
    }


def create_chart(results):
    short_names = results[
        "display_name"
    ].tolist()

    x_positions = np.arange(
        len(short_names)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.2),
    )

    width = 0.25

    for offset, metric, label, colour in [
        (
            -width,
            "precision",
            "Precision",
            "#2864A8",
        ),
        (
            0,
            "recall",
            "Recall",
            "#E07A2D",
        ),
        (
            width,
            "f1_score",
            "F1",
            "#31946B",
        ),
    ]:
        axes[0].bar(
            x_positions + offset,
            results[metric],
            width,
            label=label,
            color=colour,
        )

    axes[0].set_title(
        "Cleanlab label-error detection"
    )
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(
        x_positions,
        short_names,
        rotation=20,
        ha="right",
    )
    axes[0].legend(frameon=False)
    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    second_width = 0.34

    axes[1].bar(
        x_positions - second_width / 2,
        results[
            "matthews_correlation_coefficient"
        ],
        second_width,
        label="MCC",
        color="#7356A5",
    )

    axes[1].bar(
        x_positions + second_width / 2,
        results["average_precision"],
        second_width,
        label="Average precision",
        color="#C65472",
    )

    axes[1].set_title(
        "Class-imbalance-aware metrics"
    )
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(
        x_positions,
        short_names,
        rotation=20,
        ha="right",
    )
    axes[1].legend(frameon=False)
    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    figure.suptitle(
        "CIFAR-10N: same 5,000 images and folds",
        fontsize=14,
        fontweight="bold",
    )

    figure.tight_layout()

    figure.savefig(
        CHART_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def write_hashes(paths):
    lines = []

    for path in paths:
        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        relative_path = path.relative_to(ROOT)

        lines.append(
            f"{digest}  {relative_path}"
        )

    HASH_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing label file: {LABEL_FILE}"
        )

    if not CACHE_FILE.exists():
        raise FileNotFoundError(
            f"Missing embedding cache: {CACHE_FILE}"
        )

    source_commit = subprocess.check_output(
        [
            "git",
            "-C",
            str(SOURCE_REPOSITORY),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()

    if source_commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(
            "Unexpected CIFAR-N source commit: "
            + source_commit
        )

    human_labels = torch.load(
        LABEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    clean_labels_all = np.asarray(
        human_labels["clean_label"],
        dtype=np.int64,
    )

    with np.load(CACHE_FILE) as cache:
        subset_indices = np.asarray(
            cache["indices"],
            dtype=np.int64,
        )
        features = np.asarray(
            cache["features"],
            dtype=np.float32,
        )

    if len(subset_indices) != SUBSET_SIZE:
        raise RuntimeError(
            f"Expected {SUBSET_SIZE} cached indices, "
            f"found {len(subset_indices)}."
        )

    if features.shape != (SUBSET_SIZE, 512):
        raise RuntimeError(
            "Unexpected feature shape: "
            + str(features.shape)
        )

    clean_labels = clean_labels_all[
        subset_indices
    ]

    aggregate_labels = np.asarray(
        human_labels["aggre_label"],
        dtype=np.int64,
    )[subset_indices]

    # Generate the fold assignments once using the original
    # aggregate-label pilot, then reuse them for every label set.
    fold_generator = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    fixed_folds = list(
        fold_generator.split(
            features,
            aggregate_labels,
        )
    )

    result_records = []
    prediction_frames = []

    print(
        "CIFAR-10N FIVE-LABEL-SET COMPARISON"
    )
    print("Subset:", len(subset_indices))
    print("Features:", features.shape)
    print("Fixed folds:", len(fixed_folds))
    print("Source commit:", source_commit)
    print()

    for label_set, display_name in LABEL_SETS:
        print(f"Running {label_set}...")

        noisy_labels = np.asarray(
            human_labels[label_set],
            dtype=np.int64,
        )[subset_indices]

        actual_issue_mask = (
            noisy_labels != clean_labels
        )

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=RANDOM_SEED,
            ),
        )

        predicted_probabilities = cross_val_predict(
            classifier,
            features,
            noisy_labels,
            cv=fixed_folds,
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
        )

        suspected_issue_mask = (
            convert_to_issue_mask(
                issue_output,
                len(noisy_labels),
            )
        )

        quality_scores = get_label_quality_scores(
            labels=noisy_labels,
            pred_probs=predicted_probabilities,
        )

        metrics = calculate_detection_metrics(
            suspected_issue_mask,
            actual_issue_mask,
            quality_scores,
        )

        record = {
            "label_set": label_set,
            "display_name": display_name,
            "subset_size": SUBSET_SIZE,
            "actual_label_errors":
                int(np.sum(actual_issue_mask)),
            "suspected_label_issues":
                int(np.sum(suspected_issue_mask)),
            **metrics,
            "accuracy_against_noisy_labels":
                float(
                    accuracy_score(
                        noisy_labels,
                        predicted_labels,
                    )
                ),
            "accuracy_against_clean_labels":
                float(
                    accuracy_score(
                        clean_labels,
                        predicted_labels,
                    )
                ),
        }

        result_records.append(record)

        prediction_frames.append(
            pd.DataFrame(
                {
                    "label_set": label_set,
                    "original_index":
                        subset_indices,
                    "clean_label":
                        clean_labels,
                    "human_noisy_label":
                        noisy_labels,
                    "predicted_label":
                        predicted_labels,
                    "actual_label_error":
                        actual_issue_mask,
                    "cleanlab_flagged":
                        suspected_issue_mask,
                    "label_quality_score":
                        quality_scores,
                }
            )
        )

    results = pd.DataFrame(result_records)

    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    aggregate = results[
        results["label_set"] == "aggre_label"
    ].iloc[0]

    expected_baseline = {
        "actual_label_errors": 466,
        "suspected_label_issues": 1431,
        "true_positives": 355,
        "false_positives": 1076,
        "false_negatives": 111,
    }

    for key, expected_value in (
        expected_baseline.items()
    ):
        actual_value = int(aggregate[key])

        if actual_value != expected_value:
            raise RuntimeError(
                "Aggregate-label baseline mismatch for "
                f"{key}: expected {expected_value}, "
                f"found {actual_value}."
            )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
        float_format="%.10f",
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        float_format="%.10f",
    )

    summary = {
        "experiment":
            "CIFAR-10N five-label-set comparison",
        "source_commit": source_commit,
        "random_seed": RANDOM_SEED,
        "subset_size": SUBSET_SIZE,
        "feature_dimensions":
            int(features.shape[1]),
        "cross_validation_folds": N_FOLDS,
        "fold_definition":
            "Fixed folds stratified once using aggre_label",
        "label_sets": [
            label_set
            for label_set, _ in LABEL_SETS
        ],
        "cleanlab_version":
            cleanlab.__version__,
        "scikit_learn_version":
            sklearn.__version__,
        "pytorch_version":
            torch.__version__,
        "results":
            results.to_dict(orient="records"),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    create_chart(results)

    write_hashes(
        [
            Path(__file__).resolve(),
            RESULTS_FILE,
            PREDICTIONS_FILE,
            SUMMARY_FILE,
            CHART_FILE,
        ]
    )

    print()
    print("CIFAR-10N FIVE-LABEL-SET RESULTS")
    print(
        "Label set       Errors Flagged     P"
        "      R     F1    MCC     AP CleanAcc"
    )

    for row in results.itertuples(index=False):
        print(
            f"{row.label_set:14s} "
            f"{row.actual_label_errors:6d} "
            f"{row.suspected_label_issues:7d} "
            f"{row.precision:5.3f} "
            f"{row.recall:6.3f} "
            f"{row.f1_score:6.3f} "
            f"{row.matthews_correlation_coefficient:6.3f} "
            f"{row.average_precision:6.3f} "
            f"{row.accuracy_against_clean_labels:8.3f}"
        )

    print()
    print(
        "Aggregate-label baseline: EXACT MATCH"
    )
    print("Results:", RESULTS_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)


if __name__ == "__main__":
    main()
