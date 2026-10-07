"""Evaluate Cleanlab on a balanced 10,000-image CIFAR-100N subset."""

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
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR100
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
SUBSET_SIZE = 10000
NUMBER_OF_CLASSES = 100
N_FOLDS = 4
IMAGE_SIZE = 96
BATCH_SIZE = 128
DATALOADER_WORKERS = 0
CROSS_VALIDATION_JOBS = 1
LABEL_SET = "noisy_label"

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
    / "data/CIFAR-100_human.pt"
)
CACHE_FILE = (
    RAW_DATA
    / "cifar100n_resnet18_n10000_size96.npz"
)

RESULT_FILE = (
    RESULTS / "cifar100n_pilot_result.csv"
)
PREDICTIONS_FILE = (
    RESULTS / "cifar100n_pilot_predictions.csv"
)
SUMMARY_FILE = (
    RESULTS / "cifar100n_pilot_summary.json"
)
CHART_FILE = (
    RESULTS / "cifar100n_pilot_metrics.png"
)
HASH_FILE = (
    RESULTS / "cifar100n_pilot_hashes.txt"
)


def convert_to_issue_mask(
    issue_output,
    number_of_examples,
):
    issue_output = np.asarray(issue_output)

    if (
        issue_output.dtype == bool
        and issue_output.shape
        == (number_of_examples,)
    ):
        return issue_output

    mask = np.zeros(
        number_of_examples,
        dtype=bool,
    )

    mask[
        issue_output.astype(int)
    ] = True

    return mask


def calculate_metrics(
    suspected,
    actual,
    quality_scores,
):
    tp = int(np.sum(suspected & actual))
    fp = int(np.sum(suspected & ~actual))
    fn = int(np.sum(~suspected & actual))
    tn = int(np.sum(~suspected & ~actual))

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
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
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "matthews_correlation_coefficient":
            float(mcc),
        "average_precision":
            float(average_precision),
        "error_prevalence": prevalence,
        "precision_lift_over_prevalence":
            float(lift),
    }


def extract_or_load_features(
    dataset,
    subset_indices,
):
    if CACHE_FILE.exists():
        start = time.perf_counter()

        with np.load(CACHE_FILE) as cache:
            cached_indices = np.asarray(
                cache["indices"],
                dtype=np.int64,
            )
            features = np.asarray(
                cache["features"],
                dtype=np.float32,
            )

        if not np.array_equal(
            cached_indices,
            subset_indices,
        ):
            raise RuntimeError(
                "Cached CIFAR-100N indices differ "
                "from the deterministic subset."
            )

        if features.shape != (
            SUBSET_SIZE,
            512,
        ):
            raise RuntimeError(
                "Unexpected cached feature shape: "
                + str(features.shape)
            )

        elapsed = time.perf_counter() - start

        print(
            f"Loaded CIFAR-100N embedding cache "
            f"in {elapsed:.2f} seconds."
        )

        return features, "loaded", elapsed

    print(
        "Extracting frozen ResNet-18 features "
        "for 10,000 CIFAR-100 images..."
    )

    subset = Subset(
        dataset,
        subset_indices.tolist(),
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=False,
    )

    model = resnet18(
        weights=ResNet18_Weights.DEFAULT
    )
    model.fc = torch.nn.Identity()
    model.eval()

    batches = []
    start = time.perf_counter()
    total_batches = len(loader)

    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(
            loader,
            start=1,
        ):
            batches.append(
                model(images).cpu().numpy()
            )

            if (
                batch_number % 10 == 0
                or batch_number == total_batches
            ):
                print(
                    f"  Processed batch "
                    f"{batch_number}/{total_batches}"
                )

    features = np.concatenate(
        batches,
        axis=0,
    ).astype(np.float32)

    elapsed = time.perf_counter() - start

    if features.shape != (
        SUBSET_SIZE,
        512,
    ):
        raise RuntimeError(
            "Unexpected extracted feature shape: "
            + str(features.shape)
        )

    np.savez_compressed(
        CACHE_FILE,
        indices=subset_indices,
        features=features,
    )

    print(
        f"Saved embedding cache: {CACHE_FILE}"
    )
    print(
        f"Feature extraction time: "
        f"{elapsed:.2f} seconds"
    )

    return features, "created", elapsed


def create_chart(result):
    metric_names = [
        "Precision",
        "Recall",
        "F1",
        "MCC",
        "Average precision",
    ]

    values = [
        result["precision"],
        result["recall"],
        result["f1_score"],
        result[
            "matthews_correlation_coefficient"
        ],
        result["average_precision"],
    ]

    colours = [
        "#2864A8",
        "#E07A2D",
        "#31946B",
        "#7356A5",
        "#C65472",
    ]

    figure, axis = plt.subplots(
        figsize=(9.5, 5.4)
    )

    bars = axis.bar(
        metric_names,
        values,
        color=colours,
    )

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title(
        "CIFAR-100N label-error detection "
        "(10,000-image pilot)"
    )
    axis.grid(axis="y", alpha=0.25)

    axis.bar_label(
        bars,
        labels=[
            f"{value:.3f}"
            for value in values
        ],
        padding=3,
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

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    torch.set_num_threads(
        min(8, os.cpu_count() or 1)
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

    dataset = CIFAR100(
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

    clean_labels_all = np.asarray(
        human_labels["clean_label"],
        dtype=np.int64,
    )

    noisy_labels_all = np.asarray(
        human_labels[LABEL_SET],
        dtype=np.int64,
    )

    official_targets = np.asarray(
        dataset.targets,
        dtype=np.int64,
    )

    if not np.array_equal(
        official_targets,
        clean_labels_all,
    ):
        raise RuntimeError(
            "CIFAR-100 and CIFAR-100N "
            "image orders do not match."
        )

    all_indices = np.arange(len(dataset))

    subset_indices, _ = train_test_split(
        all_indices,
        train_size=SUBSET_SIZE,
        random_state=RANDOM_SEED,
        stratify=clean_labels_all,
    )

    subset_indices = np.sort(
        subset_indices
    )

    clean_labels = clean_labels_all[
        subset_indices
    ]

    noisy_labels = noisy_labels_all[
        subset_indices
    ]

    class_counts = np.bincount(
        clean_labels,
        minlength=NUMBER_OF_CLASSES,
    )

    if not np.all(class_counts == 100):
        raise RuntimeError(
            "The subset does not contain "
            "exactly 100 examples per class."
        )

    actual_issue_mask = (
        noisy_labels != clean_labels
    )

    if int(np.sum(actual_issue_mask)) != 3954:
        raise RuntimeError(
            "Unexpected subset disagreement count."
        )

    features, cache_status, feature_seconds = (
        extract_or_load_features(
            dataset,
            subset_indices,
        )
    )

    print()
    print(
        "Generating four-fold CIFAR-100N "
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

    if predicted_probabilities.shape != (
        SUBSET_SIZE,
        NUMBER_OF_CLASSES,
    ):
        raise RuntimeError(
            "Unexpected probability shape: "
            + str(predicted_probabilities.shape)
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

    metrics = calculate_metrics(
        suspected_issue_mask,
        actual_issue_mask,
        quality_scores,
    )

    evaluation_seconds = (
        time.perf_counter() - evaluation_start
    )

    result = {
        "dataset": "CIFAR-100N",
        "label_set": LABEL_SET,
        "subset_size": SUBSET_SIZE,
        "number_of_classes":
            NUMBER_OF_CLASSES,
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

    result_frame = pd.DataFrame([result])

    predictions = pd.DataFrame(
        {
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

    RESULT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frame.to_csv(
        RESULT_FILE,
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
            "CIFAR-100N balanced pilot",
        "source_commit": source_commit,
        "random_seed": RANDOM_SEED,
        "subset_selection":
            "10,000 examples stratified by clean fine label; exactly 100 per class",
        "fine_label_set": LABEL_SET,
        "excluded_arrays": {
            "clean_coarse_label":
                "20-class superclass labels; not comparable with 100 fine-class IDs",
            "noisy_coarse_label":
                "20-class superclass labels; requires a separate 20-class experiment",
        },
        "image_size": IMAGE_SIZE,
        "feature_dimensions":
            int(features.shape[1]),
        "cross_validation_folds": N_FOLDS,
        "cross_validation_jobs":
            CROSS_VALIDATION_JOBS,
        "dataloader_workers":
            DATALOADER_WORKERS,
        "feature_model":
            "ImageNet-pretrained ResNet-18",
        "classifier":
            "StandardScaler plus logistic regression",
        "cleanlab_version":
            cleanlab.__version__,
        "scikit_learn_version":
            sklearn.__version__,
        "pytorch_version":
            torch.__version__,
        "torchvision_version":
            torchvision.__version__,
        "result": result,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    create_chart(result)

    write_hashes(
        [
            Path(__file__).resolve(),
            RESULT_FILE,
            PREDICTIONS_FILE,
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
    print("CIFAR-100N PILOT RESULTS")
    print(f"Examples: {SUBSET_SIZE}")
    print("Classes: 100")
    print("Examples per class: 100")
    print(
        f"Actual errors: "
        f"{result['actual_label_errors']}"
    )
    print(
        f"Flagged: "
        f"{result['suspected_label_issues']}"
    )
    print(
        f"TP/FP/FN/TN: "
        f"{result['true_positives']}/"
        f"{result['false_positives']}/"
        f"{result['false_negatives']}/"
        f"{result['true_negatives']}"
    )
    print(
        f"Precision: "
        f"{result['precision']:.4f}"
    )
    print(
        f"Recall: "
        f"{result['recall']:.4f}"
    )
    print(
        f"F1: "
        f"{result['f1_score']:.4f}"
    )
    print(
        f"MCC: "
        f"{result['matthews_correlation_coefficient']:.4f}"
    )
    print(
        f"Average precision: "
        f"{result['average_precision']:.4f}"
    )
    print(
        f"Accuracy against noisy labels: "
        f"{result['accuracy_against_noisy_labels']:.4f}"
    )
    print(
        f"Accuracy against clean labels: "
        f"{result['accuracy_against_clean_labels']:.4f}"
    )

    print()
    print("RESOURCE SUMMARY")
    print(f"Feature cache: {cache_status}")
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
    print("Result:", RESULT_FILE)
    print("Predictions:", PREDICTIONS_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)


if __name__ == "__main__":
    main()
