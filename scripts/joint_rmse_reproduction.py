"""Reproduce the paper's joint-estimation RMSE on the released CIFAR-10 artifacts.

Follows the authors' work log (cifar10_work_log_experiments_tables_figures_
benchmarking.py, "Benchmarking RMSE estimating the joint"):
    true_joint = noise_matrix * p(y)
    est_joint  = estimate_joint(noisy_labels, pred_probs)
    RMSE       = sqrt(mean_squared_error(true_joint, est_joint))
for the confident joint (CL) and the argmax-confusion baseline, over the
twelve released noise/sparsity settings (20/40/70% noise; 0.0-0.6 sparsity).

Runs under either cleanlab 0.1.0 (requirements-legacy.txt, Python 3.11) or
cleanlab 2.9.0 (requirements.txt); the output file is named by version.
"""

from pathlib import Path
import csv
import hashlib
import json
import pickle

import cleanlab
import numpy as np
from sklearn.metrics import confusion_matrix, mean_squared_error

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CIFAR_ROOT = REPOSITORY_ROOT / "external" / "confidentlearning-reproduce" / "cifar10"
LABEL_DIRECTORY = CIFAR_ROOT / "cifar10_noisy_labels"
RESULTS_DIRECTORY = REPOSITORY_ROOT / "results"

CLASS_TO_INDEX = {
    "airplane": 0, "automobile": 1, "bird": 2, "cat": 3, "deer": 4,
    "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9,
}
SPARSITIES = ["0", "2", "4", "6"]
NOISE_AMOUNTS = ["2", "4", "6"]  # file names; 0.6 corresponds to ~70% noise
LEGACY = cleanlab.__version__.startswith("0.")


def estimate_joint(noisy_labels, pred_probs, confident_joint=None):
    if LEGACY:
        from cleanlab.latent_estimation import estimate_joint as fn
        return fn(noisy_labels, pred_probs, confident_joint)
    from cleanlab.count import estimate_joint as fn
    return fn(noisy_labels, pred_probs, confident_joint=confident_joint)


def load_setting(sparsity, amount):
    tag = f"frac_zero_noise_rates__0.{sparsity}__noise_amount__0.{amount}"
    with open(LABEL_DIRECTORY / f"cifar10_noisy_labels__{tag}.json") as handle:
        mapping = json.load(handle)
    noisy = np.asarray(list(mapping.values()))
    clean = np.asarray([CLASS_TO_INDEX[Path(p).parent.name] for p in mapping])
    with open(LABEL_DIRECTORY / f"cifar10_noise_matrix__{tag}.pickle", "rb") as handle:
        noise_matrix = pickle.load(handle)
    folder = CIFAR_ROOT / (
        f"cifar10_noisy_labels__frac_zero_noise_rates__0_{sparsity}"
        f"__noise_amount__0_{amount}"
    )
    pred_probs = np.load(next(folder.glob("*pyx*.npy")))[:, :10]
    if not LEGACY:  # current cleanlab validates row sums (as in Stage 3)
        pred_probs = pred_probs.astype(np.float64)
        pred_probs /= pred_probs.sum(axis=1, keepdims=True)
    return noisy, clean, noise_matrix, pred_probs


def main():
    rows = []
    for amount in NOISE_AMOUNTS:
        for sparsity in SPARSITIES:
            noisy, clean, noise_matrix, pred_probs = load_setting(sparsity, amount)
            prior = np.bincount(clean, minlength=10) / len(clean)
            true_joint = noise_matrix * prior
            cl_joint = estimate_joint(noisy, pred_probs)
            argmax_cj = confusion_matrix(np.argmax(pred_probs, axis=1), noisy).T
            argmax_joint = estimate_joint(noisy, pred_probs, argmax_cj)
            rows.append({
                "noise": round(float(np.mean(noisy != clean)), 1),
                "sparsity": int(sparsity) / 10,
                "cl_rmse": float(np.sqrt(mean_squared_error(true_joint, cl_joint))),
                "argmax_rmse": float(np.sqrt(mean_squared_error(true_joint, argmax_joint))),
            })
            r = rows[-1]
            print(f"noise={r['noise']:.1f} sparsity={r['sparsity']:.1f} "
                  f"CL RMSE={r['cl_rmse']:.4f} argmax RMSE={r['argmax_rmse']:.4f}")

    RESULTS_DIRECTORY.mkdir(exist_ok=True)
    out = RESULTS_DIRECTORY / f"joint_rmse_cleanlab_{cleanlab.__version__}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"cleanlab {cleanlab.__version__}; saved {out.name}; sha256 {digest}")


if __name__ == "__main__":
    main()
