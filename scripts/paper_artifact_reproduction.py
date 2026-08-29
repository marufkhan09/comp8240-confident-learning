"""Compatibility verification using the paper authors' released artifacts.

Configuration:
- Dataset: CIFAR-10
- Intended synthetic noise: 40%
- Noise-matrix sparsity: 60%
- Predictions: authors' four-fold ResNet-50 out-of-sample probabilities

This verifies the authors' released label-error masks and separately applies
Cleanlab 2.9.0. It does not reproduce Table 2's ten-trial retraining results.
"""

from pathlib import Path
import csv
import json
import subprocess

import cleanlab
import numpy as np
from cleanlab.filter import find_label_issues


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPOSITORY = (
    REPOSITORY_ROOT / "external" / "confidentlearning-reproduce"
)
CIFAR_ROOT = EXTERNAL_REPOSITORY / "cifar10"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

CONFIGURATION = (
    "cifar10_noisy_labels__frac_zero_noise_rates__0_6"
    "__noise_amount__0_4"
)

PROBABILITY_FILE = (
    CIFAR_ROOT
    / CONFIGURATION
    / "cifar10__train__model_resnet50__pyx.npy"
)

LABEL_FILE = (
    CIFAR_ROOT
    / "cifar10_noisy_labels"
    / (
        "cifar10_noisy_labels__frac_zero_noise_rates__0.6"
        "__noise_amount__0.4.json"
    )
)

SUMMARY_FILE = RESULTS_DIRECTORY / "paper_artifact_reproduction.json"
COMPARISON_FILE = RESULTS_DIRECTORY / "paper_artifact_comparison.csv"

CLASS_TO_INDEX = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

METHODS = {
    "argmax": {
        "filter_by": "predicted_neq_given",
        "author_directory": "train_pruned_argmax",
    },
    "cl_pbc": {
        "filter_by": "prune_by_class",
        "author_directory": "train_pruned_cl_pbc",
    },
    "cl_pbnr": {
        "filter_by": "prune_by_noise_rate",
        "author_directory": "train_pruned_cl_pbnr",
    },
    "cl_both": {
        "filter_by": "both",
        "author_directory": "train_pruned_cl_both",
    },
    "confident_joint": {
        "filter_by": "confident_learning",
        "author_directory": "train_pruned_conf_joint_only",
    },
}


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
        "flagged": int(np.sum(flagged)),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


def convert_to_mask(output, number_of_examples):
    output = np.asarray(output)

    if output.dtype == bool and output.shape == (number_of_examples,):
        return output

    mask = np.zeros(number_of_examples, dtype=bool)
    mask[output.astype(int)] = True
    return mask


def load_author_mask(directory):
    mask_file = (
        CIFAR_ROOT
        / CONFIGURATION
        / directory
        / "train_mask.npy"
    )

    keep_mask = np.load(mask_file).astype(bool)

    if keep_mask.shape != (50000,):
        raise ValueError(
            f"Unexpected author mask shape: {keep_mask.shape}"
        )

    return ~keep_mask


def get_source_commit():
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(EXTERNAL_REPOSITORY),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def main():
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not PROBABILITY_FILE.exists():
        raise FileNotFoundError(
            f"Missing probability file: {PROBABILITY_FILE}"
        )

    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Missing label file: {LABEL_FILE}")

    with LABEL_FILE.open("r", encoding="utf-8") as input_file:
        label_mapping = json.load(input_file)

    paths = list(label_mapping.keys())
    noisy_labels = np.asarray(
        list(label_mapping.values()),
        dtype=np.int64,
    )

    try:
        clean_labels = np.asarray(
            [
                CLASS_TO_INDEX[Path(image_path).parent.name]
                for image_path in paths
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Unknown CIFAR-10 class in image path: {error}"
        ) from error

    pred_probs_original = np.load(PROBABILITY_FILE)

    if pred_probs_original.shape != (50000, 10):
        raise ValueError(
            "Expected a (50000, 10) probability matrix, "
            f"found {pred_probs_original.shape}."
        )

    if len(noisy_labels) != 50000:
        raise ValueError(
            f"Expected 50000 labels, found {len(noisy_labels)}."
        )

    class_counts = np.bincount(clean_labels, minlength=10)

    if not np.array_equal(class_counts, np.full(10, 5000)):
        raise ValueError(
            f"Unexpected clean-label class counts: {class_counts}"
        )

    actual_errors = noisy_labels != clean_labels

    original_row_sums = pred_probs_original.sum(axis=1)

    # The released matrix uses float16 storage, so some rows do not sum
    # exactly to one. Convert to float64 and renormalise for the current
    # Cleanlab input validation.
    pred_probs = pred_probs_original.astype(np.float64)
    pred_probs /= pred_probs.sum(axis=1, keepdims=True)

    author_results = {}
    current_results = {}
    comparison_rows = []

    print("--- Released configuration ---")
    print("Examples:", len(noisy_labels))
    print("Actual corrupted labels:", int(actual_errors.sum()))
    print(f"Actual noise rate: {actual_errors.mean():.4f}")
    print("Probability shape:", pred_probs.shape)
    print("Released probability dtype:", pred_probs_original.dtype)
    print(
        "Released row-sum range:",
        f"{original_row_sums.min():.6f}",
        f"to {original_row_sums.max():.6f}",
    )
    print("Source commit:", get_source_commit())

    print()
    print("--- Authors' released masks ---")

    author_masks = {}

    for method, configuration in METHODS.items():
        author_flagged = load_author_mask(
            configuration["author_directory"]
        )
        author_masks[method] = author_flagged

        metrics = calculate_metrics(author_flagged, actual_errors)
        author_results[method] = metrics

        print(
            f"{method:16s} "
            f"flagged={metrics['flagged']:5d} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1_score']:.4f}"
        )

        comparison_rows.append(
            {
                "implementation": "authors_released_mask",
                "method": method,
                **metrics,
                "mask_agreement_with_author": 1.0,
                "jaccard_with_author": 1.0,
            }
        )

    print()
    print(f"--- Current Cleanlab {cleanlab.__version__} ---")

    for method, configuration in METHODS.items():
        current_output = find_label_issues(
            labels=noisy_labels,
            pred_probs=pred_probs,
            filter_by=configuration["filter_by"],
            n_jobs=2,
        )
        current_flagged = convert_to_mask(
            current_output,
            len(noisy_labels),
        )

        metrics = calculate_metrics(current_flagged, actual_errors)

        author_flagged = author_masks[method]
        mask_agreement = float(
            np.mean(current_flagged == author_flagged)
        )

        intersection = int(
            np.sum(current_flagged & author_flagged)
        )
        union = int(np.sum(current_flagged | author_flagged))
        jaccard = intersection / union if union else 1.0

        current_results[method] = {
            **metrics,
            "mask_agreement_with_author": mask_agreement,
            "jaccard_with_author": jaccard,
        }

        print(
            f"{method:16s} "
            f"flagged={metrics['flagged']:5d} "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1_score']:.4f} "
            f"agreement={mask_agreement:.4f}"
        )

        comparison_rows.append(
            {
                "implementation": (
                    f"cleanlab_{cleanlab.__version__}"
                ),
                "method": method,
                **metrics,
                "mask_agreement_with_author": mask_agreement,
                "jaccard_with_author": jaccard,
            }
        )

    summary = {
        "experiment": (
            "Compatibility verification of released paper artifacts"
        ),
        "scope_note": (
            "This is not the paper's ten-trial Table 2 retraining."
        ),
        "dataset": "CIFAR-10",
        "intended_noise_rate": 0.4,
        "noise_matrix_sparsity": 0.6,
        "actual_corrupted_labels": int(actual_errors.sum()),
        "actual_noise_rate": float(actual_errors.mean()),
        "examples": len(noisy_labels),
        "classes": 10,
        "probability_source": (
            "Authors' four-fold ResNet-50 out-of-sample predictions"
        ),
        "released_probability_dtype": str(
            pred_probs_original.dtype
        ),
        "probabilities_renormalised_for_current_cleanlab": True,
        "source_repository": (
            "https://github.com/cgnorthcutt/"
            "confidentlearning-reproduce"
        ),
        "source_commit": get_source_commit(),
        "current_cleanlab_version": cleanlab.__version__,
        "authors_released_masks": author_results,
        "current_cleanlab_results": current_results,
    }

    with SUMMARY_FILE.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    with COMPARISON_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "implementation",
                "method",
                "flagged",
                "true_positives",
                "false_positives",
                "false_negatives",
                "precision",
                "recall",
                "f1_score",
                "mask_agreement_with_author",
                "jaccard_with_author",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison_rows)

    print()
    print("Summary saved to:", SUMMARY_FILE)
    print("Comparison saved to:", COMPARISON_FILE)


if __name__ == "__main__":
    main()