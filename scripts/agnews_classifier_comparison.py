"""Compare three classifiers for AGNews-CLNoise.

Classifiers:
- TF-IDF logistic regression
- TF-IDF Complement Naive Bayes
- TF-IDF calibrated linear SVM

All classifiers use the same articles, corruptions and outer folds.
"""

from pathlib import Path
import hashlib
import json
import time

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from datasets import load_dataset
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from agnews_noise_sweep import (
    CLASS_NAMES,
    CROSS_VALIDATION_FOLDS,
    DATASET_ID,
    DATASET_REVISION,
    FOLD_SEED,
    RAW_DATA_DIRECTORY,
    SUBSET_SEED,
    SUBSET_SIZE,
    add_prefixed_metrics,
    calculate_metrics,
    convert_to_issue_mask,
    corrupt_labels,
    make_classifier,
)


NOISE_RATE = 0.20
NOISE_TYPES = ("cyclic", "symmetric")
CORRUPTION_SEEDS = (42, 43, 44)

CLASSIFIERS = (
    "logistic_regression",
    "complement_naive_bayes",
    "calibrated_linear_svm",
)

DISPLAY_NAMES = {
    "logistic_regression":
        "Logistic regression",
    "complement_naive_bayes":
        "Complement NB",
    "calibrated_linear_svm":
        "Calibrated linear SVM",
}

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIRECTORY = ROOT / "results"

RUNS_FILE = (
    RESULTS_DIRECTORY
    / "agnews_classifier_comparison_runs.csv"
)
AGGREGATE_FILE = (
    RESULTS_DIRECTORY
    / "agnews_classifier_comparison_aggregate.csv"
)
SUMMARY_FILE = (
    RESULTS_DIRECTORY
    / "agnews_classifier_comparison_summary.json"
)
CHART_FILE = (
    RESULTS_DIRECTORY
    / "agnews_classifier_comparison.png"
)
HASH_FILE = (
    RESULTS_DIRECTORY
    / "agnews_classifier_comparison_hashes.txt"
)
REFERENCE_RUNS_FILE = (
    RESULTS_DIRECTORY
    / "agnews_noise_sweep_runs.csv"
)


def make_vectorizer():
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=30_000,
        sublinear_tf=True,
    )


def build_classifier(classifier_name):
    if classifier_name == "logistic_regression":
        # Use the original function to ensure exact reproduction.
        return make_classifier(SUBSET_SEED)

    if classifier_name == "complement_naive_bayes":
        return make_pipeline(
            make_vectorizer(),
            ComplementNB(alpha=1.0),
        )

    if classifier_name == "calibrated_linear_svm":
        return make_pipeline(
            make_vectorizer(),
            CalibratedClassifierCV(
                estimator=LinearSVC(
                    C=1.0,
                    random_state=SUBSET_SEED,
                ),
                method="sigmoid",
                cv=3,
                n_jobs=1,
            ),
        )

    raise ValueError(
        f"Unknown classifier: {classifier_name}"
    )


def verify_logistic_reference(runs):
    reference = pd.read_csv(
        REFERENCE_RUNS_FILE
    )

    reference = reference[
        reference["noise_rate"].eq(NOISE_RATE)
    ].copy()

    logistic = runs[
        runs["classifier"].eq(
            "logistic_regression"
        )
    ].copy()

    merged = logistic.merge(
        reference,
        on=[
            "noise_type",
            "noise_rate",
            "corruption_seed",
        ],
        suffixes=("_new", "_reference"),
        validate="one_to_one",
    )

    if len(merged) != 6:
        raise RuntimeError(
            "Expected six matched logistic runs."
        )

    integer_metrics = [
        "injected_errors",
        "cleanlab_flagged",
        "cleanlab_true_positives",
        "cleanlab_false_positives",
        "cleanlab_false_negatives",
        "cleanlab_true_negatives",
        "argmax_flagged",
        "argmax_true_positives",
        "argmax_false_positives",
        "argmax_false_negatives",
        "argmax_true_negatives",
    ]

    float_metrics = [
        "accuracy_noisy",
        "accuracy_clean",
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

    for metric in integer_metrics:
        if not np.array_equal(
            merged[f"{metric}_new"],
            merged[f"{metric}_reference"],
        ):
            raise RuntimeError(
                f"Logistic reference mismatch: {metric}"
            )

    for metric in float_metrics:
        if not np.allclose(
            merged[f"{metric}_new"],
            merged[f"{metric}_reference"],
            rtol=0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Logistic reference mismatch: {metric}"
            )


def build_chart(aggregate):
    classifier_order = list(CLASSIFIERS)
    x_positions = np.arange(
        len(classifier_order)
    )
    width = 0.35

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
    )

    colours = {
        "cyclic": "#2864A8",
        "symmetric": "#E07A2D",
    }

    for axis, metric, title in [
        (
            axes[0],
            "cleanlab_f1_score",
            "Cleanlab F1",
        ),
        (
            axes[1],
            "cleanlab_mcc",
            "Cleanlab MCC",
        ),
    ]:
        for noise_index, noise_type in enumerate(
            NOISE_TYPES
        ):
            subset = (
                aggregate[
                    aggregate["noise_type"].eq(
                        noise_type
                    )
                ]
                .set_index("classifier")
                .loc[classifier_order]
            )

            offset = (
                -width / 2
                if noise_index == 0
                else width / 2
            )

            axis.bar(
                x_positions + offset,
                subset[f"{metric}_mean"],
                width,
                yerr=subset[f"{metric}_std"],
                capsize=4,
                color=colours[noise_type],
                label=noise_type.title(),
            )

        axis.set_xticks(
            x_positions,
            [
                DISPLAY_NAMES[name]
                for name in classifier_order
            ],
            rotation=18,
            ha="right",
        )
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Score")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)

    axes[1].legend(frameon=False)

    figure.suptitle(
        "AGNews-CLNoise classifier comparison "
        "(20% noise, mean ± SD over 3 seeds)",
        fontsize=13,
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
    total_start = time.perf_counter()

    np.random.seed(SUBSET_SEED)
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=str(RAW_DATA_DIRECTORY),
    )

    if dataset.column_names != [
        "text",
        "label",
    ]:
        raise RuntimeError(
            "Unexpected AG News columns."
        )

    if list(
        dataset.features["label"].names
    ) != CLASS_NAMES:
        raise RuntimeError(
            "Unexpected AG News class names."
        )

    all_labels = np.asarray(
        dataset["label"],
        dtype=np.int64,
    )
    all_indices = np.arange(len(dataset))

    subset_indices, _ = train_test_split(
        all_indices,
        train_size=SUBSET_SIZE,
        random_state=SUBSET_SEED,
        stratify=all_labels,
    )

    subset_indices = np.sort(
        subset_indices
    )

    subset = dataset.select(
        subset_indices.tolist()
    )

    texts = list(subset["text"])
    clean_labels = np.asarray(
        subset["label"],
        dtype=np.int64,
    )

    class_counts = np.bincount(
        clean_labels,
        minlength=len(CLASS_NAMES),
    )

    if not np.array_equal(
        class_counts,
        np.full(4, 2500),
    ):
        raise RuntimeError(
            "Unexpected balanced class counts."
        )

    reference_noisy_labels, _ = (
        corrupt_labels(
            clean_labels,
            NOISE_RATE,
            "cyclic",
            42,
        )
    )

    splitter = StratifiedKFold(
        n_splits=CROSS_VALIDATION_FOLDS,
        shuffle=True,
        random_state=FOLD_SEED,
    )

    fixed_folds = list(
        splitter.split(
            texts,
            reference_noisy_labels,
        )
    )

    run_rows = []

    total_runs = (
        len(CLASSIFIERS)
        * len(NOISE_TYPES)
        * len(CORRUPTION_SEEDS)
    )

    run_number = 0

    print(
        "AG NEWS CLASSIFIER COMPARISON"
    )
    print("Source revision:", DATASET_REVISION)
    print("Fixed subset:", SUBSET_SIZE)
    print(
        "Clean class counts:",
        class_counts.tolist(),
    )
    print("Noise rate:", NOISE_RATE)
    print("Runs:", total_runs)
    print()

    for classifier_name in CLASSIFIERS:
        for noise_type in NOISE_TYPES:
            for corruption_seed in (
                CORRUPTION_SEEDS
            ):
                run_number += 1

                run_id = (
                    f"{classifier_name}_"
                    f"{noise_type}_"
                    f"s{corruption_seed}"
                )

                print(
                    f"[{run_number:02d}/"
                    f"{total_runs:02d}] "
                    f"{run_id}"
                )

                noisy_labels, actual_errors = (
                    corrupt_labels(
                        clean_labels,
                        NOISE_RATE,
                        noise_type,
                        corruption_seed,
                    )
                )

                classifier = build_classifier(
                    classifier_name
                )

                probabilities = cross_val_predict(
                    classifier,
                    texts,
                    noisy_labels,
                    cv=fixed_folds,
                    method="predict_proba",
                    n_jobs=1,
                )

                if probabilities.shape != (
                    SUBSET_SIZE,
                    len(CLASS_NAMES),
                ):
                    raise RuntimeError(
                        "Unexpected probability shape."
                    )

                predicted_labels = np.argmax(
                    probabilities,
                    axis=1,
                )

                issue_output = find_label_issues(
                    labels=noisy_labels,
                    pred_probs=probabilities,
                    n_jobs=1,
                )

                cleanlab_mask = (
                    convert_to_issue_mask(
                        issue_output,
                        SUBSET_SIZE,
                    )
                )

                argmax_mask = (
                    predicted_labels
                    != noisy_labels
                )

                quality_scores = (
                    get_label_quality_scores(
                        labels=noisy_labels,
                        pred_probs=probabilities,
                    )
                )

                issue_scores = (
                    1.0 - quality_scores
                )

                cleanlab_metrics = (
                    calculate_metrics(
                        cleanlab_mask,
                        actual_errors,
                        issue_scores,
                    )
                )

                argmax_metrics = (
                    calculate_metrics(
                        argmax_mask,
                        actual_errors,
                    )
                )

                prevalence = float(
                    actual_errors.mean()
                )

                row = {
                    "run_id": run_id,
                    "classifier":
                        classifier_name,
                    "classifier_display":
                        DISPLAY_NAMES[
                            classifier_name
                        ],
                    "noise_type": noise_type,
                    "noise_rate": NOISE_RATE,
                    "corruption_seed":
                        corruption_seed,
                    "subset_size":
                        SUBSET_SIZE,
                    "injected_errors":
                        int(actual_errors.sum()),
                    "accuracy_noisy":
                        float(
                            accuracy_score(
                                noisy_labels,
                                predicted_labels,
                            )
                        ),
                    "accuracy_clean":
                        float(
                            accuracy_score(
                                clean_labels,
                                predicted_labels,
                            )
                        ),
                }

                add_prefixed_metrics(
                    row,
                    "cleanlab",
                    cleanlab_metrics,
                    prevalence,
                )

                add_prefixed_metrics(
                    row,
                    "argmax",
                    argmax_metrics,
                    prevalence,
                )

                run_rows.append(row)

    runs = pd.DataFrame(run_rows)

    verify_logistic_reference(runs)

    runs.to_csv(
        RUNS_FILE,
        index=False,
        lineterminator="\n",
        float_format="%.10f",
    )

    metric_columns = [
        "accuracy_noisy",
        "accuracy_clean",
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
        runs.groupby(
            [
                "classifier",
                "classifier_display",
                "noise_type",
            ],
            sort=False,
        )[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    aggregate.columns = [
        column[0]
        if not column[1]
        else (
            f"{column[0]}_{column[1]}"
        )
        for column in aggregate.columns
    ]

    aggregate.to_csv(
        AGGREGATE_FILE,
        index=False,
        lineterminator="\n",
        float_format="%.10f",
    )

    build_chart(aggregate)

    summary = {
        "experiment":
            "AGNews-CLNoise classifier comparison",
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "subset_size": SUBSET_SIZE,
        "clean_class_counts":
            class_counts.tolist(),
        "noise_rate": NOISE_RATE,
        "noise_types":
            list(NOISE_TYPES),
        "corruption_seeds":
            list(CORRUPTION_SEEDS),
        "outer_folds":
            CROSS_VALIDATION_FOLDS,
        "classifiers": {
            "logistic_regression":
                "TF-IDF plus multinomial logistic regression",
            "complement_naive_bayes":
                "TF-IDF plus ComplementNB(alpha=1.0)",
            "calibrated_linear_svm":
                "TF-IDF plus LinearSVC(C=1.0) with three-fold sigmoid calibration",
        },
        "logistic_reference_check":
            "Exact match for all six corresponding committed sweep runs",
        "cleanlab_version":
            cleanlab.__version__,
        "scikit_learn_version":
            sklearn.__version__,
        "aggregate_results":
            json.loads(
                aggregate.to_json(
                    orient="records"
                )
            ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    write_hashes(
        [
            Path(__file__).resolve(),
            RUNS_FILE,
            AGGREGATE_FILE,
            SUMMARY_FILE,
            CHART_FILE,
        ]
    )

    total_seconds = (
        time.perf_counter() - total_start
    )

    print()
    print(
        "LOGISTIC REFERENCE: "
        "EXACT MATCH (6/6 RUNS)"
    )
    print()
    print(
        "AGGREGATE CLASSIFIER RESULTS"
    )
    print(
        "Classifier             Noise"
        "      F1±SD       MCC±SD"
        "        AP    CleanAcc"
    )

    for row in aggregate.itertuples(
        index=False
    ):
        print(
            f"{row.classifier_display:22s} "
            f"{row.noise_type:9s} "
            f"{row.cleanlab_f1_score_mean:.3f}"
            f"±{row.cleanlab_f1_score_std:.3f} "
            f"{row.cleanlab_mcc_mean:.3f}"
            f"±{row.cleanlab_mcc_std:.3f} "
            f"{row.cleanlab_average_precision_mean:.3f} "
            f"{row.accuracy_clean_mean:.3f}"
        )

    print()
    print(
        f"Total runtime: "
        f"{total_seconds:.2f} seconds"
    )
    print("Runs:", RUNS_FILE)
    print("Aggregates:", AGGREGATE_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)


if __name__ == "__main__":
    main()
