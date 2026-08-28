"""
Controlled confident-learning feasibility experiment.

Purpose: produce genuine, reproducible execution evidence for the
COMP8240 project proposal (Section 3 - feasibility) using the cleanlab
package, which implements the Confident Learning framework of
Northcutt, Jiang & Chuang (2021).

Method:
  1. Generate a synthetic multi-class dataset with KNOWN true labels.
  2. Deliberately corrupt a controlled fraction of labels (a simplified class-conditional asymmetric noise process inspired by the paper's CIFAR-10 experiments).
  3. Train a simple classifier with 4-fold cross-validation to obtain
     out-of-sample predicted probabilities (as required by CL).
  4. Run cleanlab's `find_label_issues` to flag suspected label errors.
  5. Compare flagged errors against the TRUE injected corruption to compute
     precision, recall and F1 -- an exact ground-truth evaluation, since we
     control which labels were corrupted.
"""

import numpy as np
import cleanlab
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from cleanlab.filter import find_label_issues
from cleanlab.count import estimate_joint

print(f"cleanlab version: {cleanlab.__version__}")

rng = np.random.RandomState(42)

# 1. Generate synthetic classification data with true (clean) labels
N_SAMPLES = 3000
N_CLASSES = 5
X, y_true = make_classification(
    n_samples=N_SAMPLES,
    n_features=20,
    n_informative=12,
    n_classes=N_CLASSES,
    n_clusters_per_class=2,
    random_state=42,
)

# 2. Inject controlled, class-conditional label noise (20% corruption rate).
# This simplified asymmetric-noise design uses one fixed destination
# class for each source class. It is inspired by, but not identical to,
# the transition-matrix experiments reported in the paper.
FIXED_TRANSITION = {0: 2, 1: 3, 2: 4, 3: 0, 4: 1}

NOISE_RATE = 0.20
n_noisy = int(NOISE_RATE * N_SAMPLES)
noisy_idx = rng.choice(N_SAMPLES, size=n_noisy, replace=False)

y_noisy = y_true.copy()
for i in noisy_idx:
    true_class = y_true[i]
    y_noisy[i] = FIXED_TRANSITION[true_class]

is_actually_wrong = np.zeros(N_SAMPLES, dtype=bool)
is_actually_wrong[noisy_idx] = True

print(f"\nDataset: {N_SAMPLES} samples, {N_CLASSES} classes")
print(f"Injected label errors: {n_noisy} ({NOISE_RATE:.0%} of dataset)")

# 3. Get out-of-sample predicted probabilities via 4-fold cross-validation
clf = LogisticRegression(max_iter=1000, random_state=42)
pred_probs = cross_val_predict(
    clf, X, y_noisy, cv=4, method="predict_proba"
)

# 4. Run confident learning to flag suspected label errors
issue_idx = find_label_issues(
    labels=y_noisy,
    pred_probs=pred_probs,
    return_indices_ranked_by="self_confidence",
)
flagged = np.zeros(N_SAMPLES, dtype=bool)
flagged[issue_idx] = True

print(f"cleanlab flagged: {flagged.sum()} suspected label errors")

# 5. Score against ground truth
true_positives = np.sum(flagged & is_actually_wrong)
false_positives = np.sum(flagged & ~is_actually_wrong)
false_negatives = np.sum(~flagged & is_actually_wrong)

precision = true_positives / flagged.sum() if flagged.sum() > 0 else 0.0
recall = true_positives / is_actually_wrong.sum()
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print("\n--- Results (exact ground-truth evaluation) ---")
print(f"True positives (correctly flagged noisy labels): {true_positives}")
print(f"False positives (clean labels wrongly flagged): {false_positives}")
print(f"False negatives (noisy labels missed): {false_negatives}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Proportion of the dataset flagged as suspected label issues
# (NOT the same thing as a principled noise-rate estimate -- just a raw count).
proportion_flagged = flagged.sum() / N_SAMPLES

# A genuine noise-rate ESTIMATE, using cleanlab's confident-joint estimation
# (this is the CL framework's actual "counting with probabilistic thresholds"
# mechanism described in the paper, distinct from simply counting flags).
joint = estimate_joint(labels=y_noisy, pred_probs=pred_probs)
estimated_noise_rate = 1 - np.trace(joint)

print(f"\nActual injected noise rate:                  {NOISE_RATE:.4f}")
print(f"Proportion of examples flagged as issues:     {proportion_flagged:.4f}")
print(f"CL confident-joint estimated noise rate:      {estimated_noise_rate:.4f}")
