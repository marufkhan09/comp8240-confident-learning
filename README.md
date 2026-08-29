# COMP8240 Confident Learning Project

This repository supports a COMP8240 novel project based on:

> Northcutt, C. G., Jiang, L., and Chuang, I. L. (2021). *Confident Learning: Estimating Uncertainty in Dataset Labels.* Journal of Artificial Intelligence Research, 70, 1373–1411.

## Preliminary feasibility experiment

The script `scripts/cl_experiment.py` performs a controlled label-error detection experiment.

It:

1. Generates 3,000 synthetic examples across five classes.
2. Corrupts 20% of the labels using a simplified class-conditional asymmetric noise process.
3. Uses four-fold cross-validation with logistic regression to produce out-of-sample predicted probabilities.
4. Uses Cleanlab to identify suspected label errors.
5. Compares the suspected errors with the known injected errors.

This is a preliminary software feasibility test. It is not presented as a reproduction of the paper's CIFAR-10 experiments.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

```

## Run the experiment

```bash
python scripts/cl_experiment.py
```

## Codespace feasibility result

Using Python 3.12.1 and Cleanlab 2.9.0:

- Samples: 3,000
- Injected label errors: 600
- Suspected errors flagged: 1,166
- True positives: 469
- False positives: 697
- False negatives: 131
- Precision: 0.4022
- Recall: 0.7817
- F1-score: 0.5311
- True injected noise rate: 0.2000
- Confident-joint estimated noise rate: 0.5003

The experiment was executed repeatedly with identical output.

## Reproducibility files

- `requirements.txt`: exact Python dependencies
- `results/environment.txt`: software and platform information
- `results/feasibility_output.txt`: first execution output
- `results/feasibility_output_run2.txt`: repeated execution output
- `results/feasibility_output_final.txt`: output after documentation corrections


## CIFAR-10N pilot extension

This experiment applies Cleanlab to CIFAR-10N, an external dataset containing real human-provided noisy labels for the CIFAR-10 training images. It is a pilot extension and is not presented as a reproduction of the original Confident Learning paper's CIFAR-10 experiment.

### Data acquisition

The CIFAR-10N labels are obtained from the official repository:

```bash
mkdir -p external
git clone --depth 1 \
  https://github.com/UCSC-REAL/cifar-10-100n.git \
  external/cifar-10-100n
```

The experiment used repository revision:

```text
49df7d8a69e355470c77c1c2f2424916325a394b
```

The original CIFAR-10 images can be downloaded through torchvision:

```bash
mkdir -p data/raw
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='data/raw', train=True, download=True)"
```


Downloaded datasets, third-party repositories and cached image embeddings are excluded from Git because they can be recreated using these instructions.

### Method

The script `scripts/cifar10n_experiment.py`:

1. Loads the CIFAR-10 training images and CIFAR-10N human annotations.
2. Verifies that the 50,000 image labels are in exactly the same order.
3. Selects a fixed, stratified subset of 5,000 images using random seed 42.
4. Uses `aggre_label` as the human-provided noisy-label set.
5. Resizes images to 96 by 96 pixels.
6. Extracts frozen ImageNet-pretrained ResNet-18 features.
7. Uses four-fold cross-validation with logistic regression to generate out-of-sample predicted probabilities.
8. Uses Cleanlab to identify suspected label issues.
9. Compares the suspected issues with disagreements between `aggre_label` and CIFAR-10N's `clean_label`.

Run the experiment using:

```bash
python scripts/cifar10n_experiment.py \
  2>&1 | tee results/cifar10n_output.txt
```

### Pilot results

| Measure | Result |
|---|---:|
| Images in deterministic subset | 5,000 |
| Aggregate labels differing from clean reference | 466 |
| Suspected issues flagged by Cleanlab | 1,431 |
| True positives | 355 |
| False positives | 1,076 |
| False negatives | 111 |
| Precision | 0.2481 |
| Recall | 0.7618 |
| F1-score | 0.3743 |
| Accuracy against noisy labels | 0.6634 |
| Accuracy against clean reference labels | 0.6922 |

Under this evaluation definition, Cleanlab identified 355 of the 466 aggregate-label disagreements, giving recall of 0.7618. Its precision was lower because it also flagged 1,076 examples whose aggregate labels agreed with the clean reference labels.

The experiment was run twice. The generated JSON summary and all 5,000 CSV prediction records had identical SHA-256 hashes across both executions.

### Limitations

This is a preliminary CPU-feasible experiment using one deterministic 5,000-image subset, one label set, reduced image resolution, frozen ResNet-18 features and logistic regression. Its results should not be treated as a full-dataset benchmark or as a reproduction of the original paper. Further experiments should evaluate all 50,000 examples, additional CIFAR-10N label sets, alternative classifiers and multiple random seeds.

### Generated files

- `results/cifar10n_output.txt`: first execution log
- `results/cifar10n_output_run2.txt`: repeated execution log
- `results/cifar10n_summary.json`: machine-readable results
- `results/cifar10n_predictions.csv`: per-example predictions and issue decisions
- `results/cifar10n_hashes_run1.txt`: first-run artifact hashes
- `results/cifar10n_hashes_run2.txt`: repeated-run artifact hashes
- `results/cifar10n_source_commit.txt`: exact CIFAR-10N source revision


## Paper-artifact compatibility reproduction

The script `scripts/paper_artifact_reproduction.py` uses experimental artifacts released by the Confident Learning paper's authors. The selected CIFAR-10 configuration has 40% intended synthetic label noise and 60% noise-matrix sparsity.

The released artifacts include:

- 50,000 four-fold out-of-sample probability vectors generated using ResNet-50.
- Synthetic noisy labels.
- The known clean classes encoded by the ordered image paths.
- Label-pruning masks produced by the paper-era Cleanlab implementation.

This experiment verifies the authors' released masks and separately processes the same probabilities and labels using Cleanlab 2.9.0. It is not presented as a reproduction of the paper's ten-trial Table 2 classifier-retraining results.

### Obtain the authors' artifacts

```bash
mkdir -p external
git clone --depth 1 \
  https://github.com/cgnorthcutt/confidentlearning-reproduce.git \
  external/confidentlearning-reproduce
```

The experiment used repository revision:

```text
2f3155636663eb0813363dc06cd822aae6526c34
```

### Run the compatibility reproduction

```bash
python scripts/paper_artifact_reproduction.py \
  2>&1 | tee results/paper_artifact_output.txt
```

### Configuration verification

The released configuration contained 50,000 examples and 19,981 corrupted labels, corresponding to an actual noise rate of 0.3996. The predicted-probability matrix had shape 50,000 by 10 and was stored using `float16`.

Because float16 rounding caused some probability rows to sum to slightly less than one, the compatibility script converts the probabilities to float64 and renormalises each row before using the current Cleanlab API. The authors' released masks are evaluated without alteration.

### Results

| Method | Authors' mask F1 | Cleanlab 2.9 F1 | Mask agreement |
|---|---:|---:|---:|
| Argmax disagreement | 0.7880 | 0.7879 | 0.9998 |
| Prune by class | 0.7946 | 0.7945 | 0.9996 |
| Prune by noise rate | 0.8004 | 0.8002 | 0.9997 |
| Both pruning methods | 0.7830 | 0.7829 | 0.9995 |
| Confident joint | 0.8025 | 0.7900 | 0.9851 |

Four methods produced more than 99.95% agreement between the paper-era released masks and Cleanlab 2.9.0. The confident-joint method produced 98.51% agreement. This indicates that the released artifacts remain highly compatible with the current implementation, while small version-related differences remain and should be reported rather than treated as an exact reproduction.

The experiment was run twice, producing identical JSON and CSV SHA-256 hashes.

### Generated files

- `results/paper_artifact_output.txt`: first execution log
- `results/paper_artifact_output_run2.txt`: repeated execution log
- `results/paper_artifact_reproduction.json`: complete results and metadata
- `results/paper_artifact_comparison.csv`: method-level comparison
- `results/paper_artifact_hashes_run1.txt`: first-run hashes
- `results/paper_artifact_hashes_run2.txt`: repeated-run hashes