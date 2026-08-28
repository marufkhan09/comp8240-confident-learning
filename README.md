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
