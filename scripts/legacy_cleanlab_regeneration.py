"""Regenerate the authors' released CIFAR-10 label-error masks with cleanlab 0.1.0.

The authors' artifact README states that cleanlab v0.1.0 was used for the
original Confident Learning paper. This script runs the exact calls from the
authors' work log (cifar10_work_log_experiments_tables_figures_benchmarking.py,
commit 2f31556) on the released, unmodified float16 ResNet-50 probabilities for
all 12 released noise/sparsity settings, and compares every regenerated mask
with the corresponding released train_mask.npy.

It must run in a separate legacy environment because cleanlab 0.1.0 predates
current NumPy and scikit-learn:

    uv venv -p 3.11 .venv-legacy
    VIRTUAL_ENV=.venv-legacy uv pip install -r requirements-legacy.txt
    .venv-legacy/bin/python scripts/legacy_cleanlab_regeneration.py

(Plain `python3.11 -m venv .venv-legacy` and pip work equally well.)
"""

from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

import cleanlab  # noqa: E402
import cleanlab.pruning  # noqa: E402
from cleanlab import baseline_methods  # noqa: E402
from cleanlab.latent_estimation import compute_confident_joint  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CIFAR_ROOT = REPOSITORY_ROOT / "external" / "confidentlearning-reproduce" / "cifar10"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"
RESULTS_FILE = RESULTS_DIRECTORY / "legacy_cleanlab_regeneration.csv"
SUMMARY_FILE = RESULTS_DIRECTORY / "legacy_cleanlab_regeneration_summary.json"
HASHES_FILE = RESULTS_DIRECTORY / "legacy_cleanlab_regeneration_hashes.txt"

EXPECTED_CLEANLAB_VERSION = "0.1.0"
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
# Folder digits -> (intended noise, sparsity). Folder noise "6" yields ~70%
# actual noise, matching the paper's 20/40/70% noise settings.
NOISE_DIGITS = (2, 4, 6)
SPARSITY_DIGITS = (0, 2, 4, 6)
METHODS = ("conf_joint_only", "argmax", "cl_pbc", "cl_pbnr", "cl_both")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def detection_metrics(flagged, actual_errors):
    tp = int((flagged & actual_errors).sum())
    fp = int((flagged & ~actual_errors).sum())
    fn = int((~flagged & actual_errors).sum())
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall, 2 * precision * recall / (precision + recall)


def regenerate(noisy, probabilities):
    """The authors' calls, unchanged (work log lines 257-275)."""
    conf_joint = np.zeros(len(noisy), dtype=bool)
    conf_joint[
        compute_confident_joint(
            noisy, probabilities, return_indices_of_off_diagonals=True
        )[1]
    ] = True
    return {
        "conf_joint_only": conf_joint,
        "argmax": baseline_methods.baseline_argmax(probabilities, noisy),
        "cl_pbc": cleanlab.pruning.get_noise_indices(
            noisy, probabilities, prune_method="prune_by_class"
        ),
        "cl_pbnr": cleanlab.pruning.get_noise_indices(
            noisy, probabilities, prune_method="prune_by_noise_rate"
        ),
        "cl_both": cleanlab.pruning.get_noise_indices(
            noisy, probabilities, prune_method="both"
        ),
    }


def main():
    if cleanlab.__version__ != EXPECTED_CLEANLAB_VERSION:
        raise SystemExit(
            f"Expected cleanlab {EXPECTED_CLEANLAB_VERSION}, found "
            f"{cleanlab.__version__}. Use the legacy environment."
        )
    if not CIFAR_ROOT.exists():
        raise SystemExit(f"Missing authors' artifacts: {CIFAR_ROOT}")

    rows = []
    print(f"cleanlab {cleanlab.__version__}, NumPy {np.__version__}, "
          f"Python {sys.version.split()[0]}")
    print(f"{'setting':10s}{'method':17s}{'released F1':>12s}"
          f"{'regen F1':>10s}{'diff ex.':>9s}")

    for noise_digit in NOISE_DIGITS:
        for sparsity_digit in SPARSITY_DIGITS:
            folder = CIFAR_ROOT / (
                f"cifar10_noisy_labels__frac_zero_noise_rates__0_{sparsity_digit}"
                f"__noise_amount__0_{noise_digit}"
            )
            label_file = CIFAR_ROOT / "cifar10_noisy_labels" / (
                f"cifar10_noisy_labels__frac_zero_noise_rates__0.{sparsity_digit}"
                f"__noise_amount__0.{noise_digit}.json"
            )
            mapping = json.loads(label_file.read_text(encoding="utf-8"))
            noisy = np.asarray(list(mapping.values()))
            clean = np.asarray(
                [CLASSES.index(Path(path).parent.name) for path in mapping]
            )
            actual_errors = noisy != clean
            probabilities = np.load(
                folder / "cifar10__train__model_resnet50__pyx.npy"
            )[:, :10]

            regenerated = regenerate(noisy, probabilities)
            for method in METHODS:
                released = ~np.load(
                    folder / f"train_pruned_{method}" / "train_mask.npy"
                )
                regen = np.asarray(regenerated[method], dtype=bool)
                rel_p, rel_r, rel_f1 = detection_metrics(released, actual_errors)
                gen_p, gen_r, gen_f1 = detection_metrics(regen, actual_errors)
                differing = int((released != regen).sum())
                row = {
                    "actual_noise_rate": round(float(actual_errors.mean()), 4),
                    "sparsity": sparsity_digit / 10,
                    "method": method,
                    "released_precision": round(rel_p, 4),
                    "released_recall": round(rel_r, 4),
                    "released_f1": round(rel_f1, 4),
                    "regenerated_precision": round(gen_p, 4),
                    "regenerated_recall": round(gen_r, 4),
                    "regenerated_f1": round(gen_f1, 4),
                    "examples_differing": differing,
                    "identical_mask": differing == 0,
                }
                rows.append(row)
                setting = f"n{round(actual_errors.mean() * 100)}_s{sparsity_digit * 10}"
                print(f"{setting:10s}{method:17s}{rel_f1:12.4f}"
                      f"{gen_f1:10.4f}{differing:9d}")

    # Control: the modern cleanlab path needs float64-renormalised rows.
    # Measure how much that conversion alone changes the original method.
    folder = CIFAR_ROOT / (
        "cifar10_noisy_labels__frac_zero_noise_rates__0_6__noise_amount__0_4"
    )
    label_file = CIFAR_ROOT / "cifar10_noisy_labels" / (
        "cifar10_noisy_labels__frac_zero_noise_rates__0.6__noise_amount__0.4.json"
    )
    mapping = json.loads(label_file.read_text(encoding="utf-8"))
    noisy = np.asarray(list(mapping.values()))
    clean = np.asarray([CLASSES.index(Path(path).parent.name) for path in mapping])
    probabilities = np.load(folder / "cifar10__train__model_resnet50__pyx.npy")[:, :10]
    renormalised = probabilities.astype(np.float64)
    renormalised /= renormalised.sum(axis=1, keepdims=True)
    renorm_mask = regenerate(noisy, renormalised)["conf_joint_only"]
    released = ~np.load(folder / "train_pruned_conf_joint_only" / "train_mask.npy")
    _, _, renorm_f1 = detection_metrics(renorm_mask, noisy != clean)
    renorm_control = {
        "setting": "40% noise, 60% sparsity, confident joint",
        "input": "float64 renormalised probabilities (as required by cleanlab 2.9.0)",
        "f1": round(renorm_f1, 4),
        "examples_differing_from_released": int((renorm_mask != released).sum()),
    }
    print("Renormalisation control (cleanlab 0.1.0, confident joint, 40%/60%):",
          f"F1={renorm_f1:.4f},",
          renorm_control["examples_differing_from_released"],
          "examples differ from released mask")

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    with RESULTS_FILE.open("w", encoding="utf-8") as file:
        header = list(rows[0])
        file.write(",".join(header) + "\n")
        for row in rows:
            file.write(",".join(str(row[key]) for key in header) + "\n")

    identical = sum(row["identical_mask"] for row in rows)
    summary = {
        "experiment": "Regenerate released CIFAR-10 masks with cleanlab 0.1.0",
        "source_commit": "2f3155636663eb0813363dc06cd822aae6526c34",
        "settings": 12,
        "masks_compared": len(rows),
        "identical_masks": identical,
        "confident_joint_identical_in_all_settings": all(
            row["identical_mask"] for row in rows
            if row["method"] == "conf_joint_only"
        ),
        "max_examples_differing_in_any_mask": max(
            row["examples_differing"] for row in rows
        ),
        "max_abs_f1_difference": round(max(
            abs(row["released_f1"] - row["regenerated_f1"]) for row in rows
        ), 4),
        "probabilities": "released float16 values, unmodified",
        "renormalisation_control": renorm_control,
        "environment": {
            "python": sys.version.split()[0],
            "cleanlab": cleanlab.__version__,
            "numpy": np.__version__,
        },
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with HASHES_FILE.open("w", encoding="utf-8") as file:
        for path in (RESULTS_FILE, SUMMARY_FILE):
            file.write(f"{sha256_file(path)}  {path.relative_to(REPOSITORY_ROOT)}\n")

    print()
    print(f"Identical masks: {identical}/{len(rows)}")
    print("Max examples differing in any mask:",
          summary["max_examples_differing_in_any_mask"], "of 50,000")
    for path in (RESULTS_FILE, SUMMARY_FILE, HASHES_FILE):
        print("Saved:", path.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
