"""Run Cleanlab on all 50,000 CIFAR-10N aggregate labels."""

from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
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
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)
from torchvision.transforms import (
    Compose,
    Normalize,
    Resize,
    ToTensor,
)


RANDOM_SEED = 42
N_FOLDS = 4
IMAGE_SIZE = 96
BATCH_SIZE = 128
DATALOADER_WORKERS = 0
CROSS_VALIDATION_JOBS = 1
LABEL_SET = "aggre_label"
EXPECTED_EXAMPLES = 50000

EXPECTED_SOURCE_COMMIT = (
    "49df7d8a69e355470c77c1c2f2424916325a394b"
)

ROOT = Path(__file__).resolve().parents[1]

RAW_DATA = ROOT / "data/raw"
RESULTS = ROOT / "results"

SOURCE_REPOSITORY = (
    ROOT / "external/cifar-10-100n"
)
LABEL_FILE = (
    SOURCE_REPOSITORY
    / "data/CIFAR-10_human.pt"
)

CACHE_FILE = (
    RAW_DATA
    / "cifar10n_resnet18_n50000_size96.npz"
)

PILOT_RESULTS_FILE = (
    RESULTS / "cifar10n_label_sets_results.csv"
)

FULL_RESULTS_FILE = (
    RESULTS / "cifar10n_full_results.csv"
)
PREDICTIONS_FILE = (
    RESULTS / "cifar10n_full_predictions.csv"
)
COMPARISON_FILE = (
    RESULTS / "cifar10n_full_vs_pilot.csv"
)
SUMMARY_FILE = (
    RESULTS / "cifar10n_full_summary.json"
)
CHART_FILE = (
    RESULTS / "cifar10n_full_vs_pilot.png"
)
HASH_FILE = (
    RESULTS / "cifar10n_full_hashes.txt"
)


def convert_to_issue_mask(
    issue_output,
    number_of_examples,
):
    issue_output = np.asarray(issue_output)

    if (
        issue_output.dtype == bool
        and issue_output.shape == (
            number_of_examples,
        )
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


def calculate_metrics(
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

    average_precision = (
        average_precision_score(
            actual.astype(int),
            1.0 - quality_scores,
        )
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
        "average_precision":
            float(average_precision),
        "error_prevalence": prevalence,
        "precision_lift_over_prevalence":
            float(lift),
    }


def extract_or_load_features(dataset):
    if CACHE_FILE.exists():
        start = time.perf_counter()

        with np.load(CACHE_FILE) as cache:
            indices = np.asarray(
                cache["indices"],
                dtype=np.int64,
            )
            features = np.asarray(
                cache["features"],
                dtype=np.float32,
            )

        elapsed = time.perf_counter() - start

        if not np.array_equal(
            indices,
            np.arange(EXPECTED_EXAMPLES),
        ):
            raise RuntimeError(
                "Full cache indices are incorrect."
            )

        if features.shape != (
            EXPECTED_EXAMPLES,
            512,
        ):
            raise RuntimeError(
                "Unexpected cached feature shape: "
                + str(features.shape)
            )

        print(
            f"Loaded full embedding cache "
            f"in {elapsed:.2f} seconds."
        )

        return features, "loaded", elapsed

    print(
        "Extracting frozen ResNet-18 features "
        "for 50,000 images..."
    )

    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=False,
    )

    feature_batches = []
    start = time.perf_counter()

    total_batches = len(loader)

    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(
            loader,
            start=1,
        ):
            feature_batches.append(
                model(images).cpu().numpy()
            )

            if (
                batch_number % 25 == 0
                or batch_number == total_batches
            ):
                print(
                    f"  Processed batch "
                    f"{batch_number}/{total_batches}"
                )

    features = np.concatenate(
        feature_batches,
        axis=0,
    ).astype(np.float32)

    elapsed = time.perf_counter() - start

    if features.shape != (
        EXPECTED_EXAMPLES,
        512,
    ):
        raise RuntimeError(
            "Unexpected extracted feature shape: "
            + str(features.shape)
        )

    indices = np.arange(
        EXPECTED_EXAMPLES,
        dtype=np.int64,
    )

    np.savez_compressed(
        CACHE_FILE,
        indices=indices,
        features=features,
    )

    print(
        f"Saved full embedding cache: "
        f"{CACHE_FILE}"
    )
    print(
        f"Feature extraction time: "
        f"{elapsed:.2f} seconds"
    )

    return features, "created", elapsed


def create_comparison_chart(comparison):
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1"),
        (
            "matthews_correlation_coefficient",
            "MCC",
        ),
        ("average_precision", "AP"),
    ]

    positions = np.arange(len(metrics))
    width = 0.36

    pilot = comparison[
        comparison["scope"] == "Pilot 5,000"
    ].iloc[0]

    full = comparison[
        comparison["scope"] == "Full 50,000"
    ].iloc[0]

    pilot_values = [
        pilot[column]
        for column, _ in metrics
    ]
    full_values = [
        full[column]
        for column, _ in metrics
    ]

    figure, axis = plt.subplots(
        figsize=(10.5, 5.6)
    )

    axis.bar(
        positions - width / 2,
        pilot_values,
        width,
        label="Pilot 5,000",
        color="#7699C7",
    )

    axis.bar(
        positions + width / 2,
        full_values,
        width,
        label="Full 50,000",
        color="#2864A8",
    )

    axis.set_xticks(
        positions,
        [
            display_name
            for _, display_name in metrics
        ],
    )

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(
        "CIFAR-10N aggregate labels: "
        "pilot versus full dataset"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

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

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    available_cpus = os.cpu_count() or 1
    torch.set_num_threads(
        min(8, available_cpus)
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

    transform = Compose(
        [
            Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                antialias=True,
            ),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = CIFAR10(
        root=str(RAW_DATA),
        train=True,
        download=False,
        transform=transform,
    )

    human_labels = torch.load(
        LABEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    clean_labels = np.asarray(
        human_labels["clean_label"],
        dtype=np.int64,
    )

    noisy_labels = np.asarray(
        human_labels[LABEL_SET],
        dtype=np.int64,
    )

    official_targets = np.asarray(
        dataset.targets,
        dtype=np.int64,
    )

    if not np.array_equal(
        official_targets,
        clean_labels,
    ):
        raise RuntimeError(
            "CIFAR-10 and CIFAR-10N "
            "image orders do not match."
        )

    if len(dataset) != EXPECTED_EXAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_EXAMPLES} images, "
            f"found {len(dataset)}."
        )

    features, cache_status, feature_seconds = (
        extract_or_load_features(dataset)
    )

    print()
    print(
        "Generating four-fold full-dataset "
        "out-of-sample probabilities..."
    )

    evaluation_start = time.perf_counter()

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_SEED,
        ),
    )

    cross_validation = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    predicted_probabilities = cross_val_predict(
        classifier,
        features,
        noisy_labels,
        cv=cross_validation,
        method="predict_proba",
        n_jobs=CROSS_VALIDATION_JOBS,
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

    actual_issue_mask = (
        noisy_labels != clean_labels
    )

    metrics = calculate_metrics(
        suspected_issue_mask,
        actual_issue_mask,
        quality_scores,
    )

    evaluation_seconds = (
        time.perf_counter() - evaluation_start
    )

    full_record = {
        "scope": "Full 50,000",
        "examples": EXPECTED_EXAMPLES,
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

    full_results = pd.DataFrame(
        [full_record]
    )

    predictions = pd.DataFrame(
        {
            "original_index":
                np.arange(EXPECTED_EXAMPLES),
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

    pilot_results = pd.read_csv(
        PILOT_RESULTS_FILE
    )

    pilot = pilot_results[
        pilot_results["label_set"]
        == "aggre_label"
    ].iloc[0]

    comparison_columns = [
        "scope",
        "examples",
        "actual_label_errors",
        "suspected_label_issues",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "precision",
        "recall",
        "f1_score",
        "matthews_correlation_coefficient",
        "average_precision",
        "accuracy_against_noisy_labels",
        "accuracy_against_clean_labels",
    ]

    pilot_record = {
        "scope": "Pilot 5,000",
        "examples": 5000,
    }

    for column in comparison_columns[2:]:
        pilot_record[column] = pilot[column]

    comparison = pd.DataFrame(
        [
            pilot_record,
            {
                column: full_record[column]
                for column in comparison_columns
            },
        ]
    )

    FULL_RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_results.to_csv(
        FULL_RESULTS_FILE,
        index=False,
        float_format="%.10f",
    )

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
        float_format="%.10f",
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False,
        float_format="%.10f",
    )

    summary = {
        "experiment":
            "CIFAR-10N full aggregate-label run",
        "source_commit": source_commit,
        "random_seed": RANDOM_SEED,
        "examples": EXPECTED_EXAMPLES,
        "image_size": IMAGE_SIZE,
        "feature_dimensions":
            int(features.shape[1]),
        "cross_validation_folds": N_FOLDS,
        "cross_validation_jobs":
            CROSS_VALIDATION_JOBS,
        "batch_size": BATCH_SIZE,
        "dataloader_workers":
            DATALOADER_WORKERS,
        "feature_model":
            "ImageNet-pretrained ResNet-18",
        "classifier":
            "StandardScaler plus logistic regression",
        "label_set": LABEL_SET,
        "cleanlab_version":
            cleanlab.__version__,
        "scikit_learn_version":
            sklearn.__version__,
        "pytorch_version":
            torch.__version__,
        "torchvision_version":
            torchvision.__version__,
        "full_result": full_record,
        "pilot_comparison":
            comparison.to_dict(
                orient="records"
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

    create_comparison_chart(comparison)

    write_hashes(
        [
            Path(__file__).resolve(),
            FULL_RESULTS_FILE,
            PREDICTIONS_FILE,
            COMPARISON_FILE,
            SUMMARY_FILE,
            CHART_FILE,
        ]
    )

    total_seconds = (
        time.perf_counter() - total_start
    )

    cache_size_mib = (
        CACHE_FILE.stat().st_size
        / 1024**2
    )

    peak_memory_mib = (
        resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        / 1024
    )

    print()
    print("CIFAR-10N FULL-DATASET RESULTS")
    print(
        f"Examples: "
        f"{full_record['examples']}"
    )
    print(
        f"Actual errors: "
        f"{full_record['actual_label_errors']}"
    )
    print(
        f"Flagged: "
        f"{full_record['suspected_label_issues']}"
    )
    print(
        f"TP/FP/FN/TN: "
        f"{full_record['true_positives']}/"
        f"{full_record['false_positives']}/"
        f"{full_record['false_negatives']}/"
        f"{full_record['true_negatives']}"
    )
    print(
        f"Precision: "
        f"{full_record['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{full_record['recall']:.4f}"
    )
    print(
        f"F1: "
        f"{full_record['f1_score']:.4f}"
    )
    print(
        f"MCC: "
        f"{full_record['matthews_correlation_coefficient']:.4f}"
    )
    print(
        f"Average precision: "
        f"{full_record['average_precision']:.4f}"
    )
    print(
        f"Accuracy against noisy labels: "
        f"{full_record['accuracy_against_noisy_labels']:.4f}"
    )
    print(
        f"Accuracy against clean labels: "
        f"{full_record['accuracy_against_clean_labels']:.4f}"
    )

    print()
    print("PILOT VERSUS FULL")
    print(
        comparison[
            [
                "scope",
                "examples",
                "actual_label_errors",
                "suspected_label_issues",
                "precision",
                "recall",
                "f1_score",
                "matthews_correlation_coefficient",
                "average_precision",
            ]
        ].to_string(index=False)
    )

    print()
    print("RESOURCE SUMMARY")
    print(
        f"Feature cache: {cache_status}"
    )
    print(
        f"Feature stage: "
        f"{feature_seconds:.2f} seconds"
    )
    print(
        f"Evaluation stage: "
        f"{evaluation_seconds:.2f} seconds"
    )
    print(
        f"Total runtime: "
        f"{total_seconds:.2f} seconds"
    )
    print(
        f"Cache size: "
        f"{cache_size_mib:.2f} MiB"
    )
    print(
        f"Peak process memory: "
        f"{peak_memory_mib:.2f} MiB"
    )

    print()
    print("Results:", FULL_RESULTS_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Comparison:", COMPARISON_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)


if __name__ == "__main__":
    main()
