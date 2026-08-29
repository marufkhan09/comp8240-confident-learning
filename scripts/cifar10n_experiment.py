"""Pilot Confident Learning experiment on CIFAR-10N.

This is an extension experiment using real human-annotated noisy labels.
It is not a reproduction of the original paper's CIFAR-10 experiment.

Method:
1. Select a deterministic, stratified subset of 5,000 CIFAR-10 images.
2. Extract frozen pretrained ResNet-18 image features.
3. Train logistic regression using four-fold cross-validation.
4. Generate out-of-sample predicted probabilities.
5. Use Cleanlab to identify suspected label issues.
6. Compare them with verified CIFAR-10N label disagreements.
"""

from pathlib import Path
import csv
import json
import random

import cleanlab
import numpy as np
import sklearn
import torch
import torchvision
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


RANDOM_SEED = 42
N_SUBSET = 5000
N_FOLDS = 4
IMAGE_SIZE = 96
BATCH_SIZE = 64
LABEL_SET = "aggre_label"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIRECTORY = REPOSITORY_ROOT / "data" / "raw"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

LABEL_FILE = (
    REPOSITORY_ROOT
    / "external"
    / "cifar-10-100n"
    / "data"
    / "CIFAR-10_human.pt"
)

EMBEDDING_CACHE = (
    RAW_DATA_DIRECTORY
    / f"cifar10n_resnet18_n{N_SUBSET}_size{IMAGE_SIZE}.npz"
)

SUMMARY_FILE = RESULTS_DIRECTORY / "cifar10n_summary.json"
PREDICTIONS_FILE = RESULTS_DIRECTORY / "cifar10n_predictions.csv"


def extract_features(dataset, subset_indices):
    """Extract frozen ResNet-18 embeddings for the selected images."""

    if EMBEDDING_CACHE.exists():
        cached = np.load(EMBEDDING_CACHE)
        cached_indices = cached["indices"]
        cached_features = cached["features"]

        if np.array_equal(cached_indices, subset_indices):
            print(f"Loaded cached embeddings: {EMBEDDING_CACHE}")
            return cached_features

        raise RuntimeError(
            "The cached embedding indices do not match the selected subset."
        )

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()

    subset = Subset(dataset, subset_indices.tolist())
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
    )

    feature_batches = []

    print("Extracting frozen ResNet-18 embeddings...")

    with torch.inference_mode():
        for batch_number, (images, _) in enumerate(loader, start=1):
            embeddings = model(images)
            feature_batches.append(embeddings.cpu().numpy())

            if batch_number % 10 == 0 or batch_number == len(loader):
                print(f"  Processed batch {batch_number}/{len(loader)}")

    features = np.concatenate(feature_batches, axis=0)

    np.savez_compressed(
        EMBEDDING_CACHE,
        indices=subset_indices,
        features=features,
    )

    print(f"Saved embedding cache: {EMBEDDING_CACHE}")
    return features


def convert_to_issue_mask(issue_output, number_of_examples):
    """Support either boolean-mask or ranked-index Cleanlab output."""

    issue_output = np.asarray(issue_output)

    if (
        issue_output.dtype == bool
        and issue_output.shape == (number_of_examples,)
    ):
        return issue_output

    issue_mask = np.zeros(number_of_examples, dtype=bool)
    issue_mask[issue_output.astype(int)] = True
    return issue_mask


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(2)

    RAW_DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not LABEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-10N label file: {LABEL_FILE}"
        )

    transform = Compose(
        [
            Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
            ToTensor(),
            Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = CIFAR10(
        root=str(RAW_DATA_DIRECTORY),
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
    official_targets = np.asarray(dataset.targets, dtype=np.int64)

    if not np.array_equal(official_targets, clean_labels_all):
        raise RuntimeError(
            "CIFAR-10 image order does not match CIFAR-10N label order."
        )

    all_indices = np.arange(len(dataset))

    subset_indices, _ = train_test_split(
        all_indices,
        train_size=N_SUBSET,
        random_state=RANDOM_SEED,
        stratify=noisy_labels_all,
    )
    subset_indices = np.sort(subset_indices)

    clean_labels = clean_labels_all[subset_indices]
    noisy_labels = noisy_labels_all[subset_indices]
    actual_issue_mask = noisy_labels != clean_labels

    features = extract_features(dataset, subset_indices)

    print("Generating four-fold out-of-sample probabilities...")

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
        n_jobs=2,
    )

    predicted_labels = np.argmax(predicted_probabilities, axis=1)

    issue_output = find_label_issues(
        labels=noisy_labels,
        pred_probs=predicted_probabilities,
    )
    suspected_issue_mask = convert_to_issue_mask(
        issue_output,
        len(noisy_labels),
    )

    label_quality_scores = get_label_quality_scores(
        labels=noisy_labels,
        pred_probs=predicted_probabilities,
    )

    true_positives = int(
        np.sum(suspected_issue_mask & actual_issue_mask)
    )
    false_positives = int(
        np.sum(suspected_issue_mask & ~actual_issue_mask)
    )
    false_negatives = int(
        np.sum(~suspected_issue_mask & actual_issue_mask)
    )

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

    noisy_accuracy = accuracy_score(noisy_labels, predicted_labels)
    clean_accuracy = accuracy_score(clean_labels, predicted_labels)

    summary = {
        "experiment_type": "CIFAR-10N pilot extension",
        "label_set": LABEL_SET,
        "random_seed": RANDOM_SEED,
        "subset_size": N_SUBSET,
        "image_size": IMAGE_SIZE,
        "cross_validation_folds": N_FOLDS,
        "actual_label_errors": int(np.sum(actual_issue_mask)),
        "suspected_label_issues": int(np.sum(suspected_issue_mask)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy_against_noisy_labels": float(noisy_accuracy),
        "accuracy_against_clean_labels": float(clean_accuracy),
        "cleanlab_version": cleanlab.__version__,
        "scikit_learn_version": sklearn.__version__,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    with PREDICTIONS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(
            [
                "original_index",
                "clean_label",
                "human_noisy_label",
                "predicted_label",
                "actual_label_error",
                "cleanlab_flagged",
                "label_quality_score",
            ]
        )

        for row_number, original_index in enumerate(subset_indices):
            writer.writerow(
                [
                    int(original_index),
                    int(clean_labels[row_number]),
                    int(noisy_labels[row_number]),
                    int(predicted_labels[row_number]),
                    bool(actual_issue_mask[row_number]),
                    bool(suspected_issue_mask[row_number]),
                    float(label_quality_scores[row_number]),
                ]
            )

    print()
    print("--- CIFAR-10N pilot-extension results ---")
    print(f"Subset size: {N_SUBSET}")
    print(f"Human label set: {LABEL_SET}")
    print(f"Actual human-label errors: {np.sum(actual_issue_mask)}")
    print(
        f"Cleanlab suspected issues: "
        f"{np.sum(suspected_issue_mask)}"
    )
    print(f"True positives:  {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1_score:.4f}")
    print(f"Accuracy against noisy labels: {noisy_accuracy:.4f}")
    print(f"Accuracy against clean labels: {clean_accuracy:.4f}")
    print()
    print(f"Summary saved to: {SUMMARY_FILE}")
    print(f"Predictions saved to: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()