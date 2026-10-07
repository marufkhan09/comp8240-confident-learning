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

### Five-label-set comparison

The follow-up script `scripts/cifar10n_label_sets_experiment.py` compares all five official CIFAR-10N human-label sets: `aggre_label`, `worse_label` and the three individual annotator sets. Every comparison uses the same 5,000 images, cached 512-dimensional ResNet-18 features and four cross-validation folds. The folds are defined once using the aggregate labels and then held fixed across all label sets.

| Label set | Actual errors | Flagged | Precision | Recall | F1 | MCC | Average precision | Clean-label accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Aggregate | 466 | 1,431 | 0.248 | 0.762 | 0.374 | 0.337 | 0.427 | 0.692 |
| Worst-case | 2,023 | 2,879 | 0.565 | 0.805 | 0.664 | 0.382 | 0.708 | 0.470 |
| Annotator 1 | 874 | 2,066 | 0.342 | 0.808 | 0.480 | 0.369 | 0.564 | 0.601 |
| Annotator 2 | 908 | 2,094 | 0.359 | 0.827 | 0.500 | 0.390 | 0.581 | 0.604 |
| Annotator 3 | 905 | 2,205 | 0.348 | 0.849 | 0.494 | 0.386 | 0.560 | 0.584 |

The aggregate-label result exactly reproduces the earlier pilot counts. Recall remained between 0.762 and 0.849 across the five label sets. The worst-case set produced the highest F1 but also had a much higher error prevalence of 40.46%, compared with 9.32% for the aggregate labels. Consequently, its higher precision and F1 should not be interpreted by themselves as evidence of better detection. MCC remained within a narrower range of 0.337 to 0.390. Accuracy against the clean reference labels declined to 0.470 for the worst-case labels, compared with 0.692 for the aggregate labels.

### Limitations

This remains a CPU-feasible experiment using one deterministic 5,000-image subset, reduced image resolution, frozen ResNet-18 features and logistic regression. All five human-label sets are now evaluated under fixed folds, but the results should not be treated as a full-dataset benchmark or as a reproduction of the original paper. Further work should evaluate all 50,000 images, alternative classifiers and multiple subset or training seeds.

### Generated files

- `results/cifar10n_output.txt`: first execution log
- `results/cifar10n_output_run2.txt`: repeated execution log
- `results/cifar10n_summary.json`: machine-readable results
- `results/cifar10n_predictions.csv`: per-example predictions and issue decisions
- `results/cifar10n_hashes_run1.txt`: first-run artifact hashes
- `results/cifar10n_hashes_run2.txt`: repeated-run artifact hashes
- `results/cifar10n_source_commit.txt`: exact CIFAR-10N source revision
- `scripts/cifar10n_label_sets_experiment.py`: fixed-subset five-label-set comparison
- `results/cifar10n_label_sets_results.csv`: label-set-level metrics
- `results/cifar10n_label_sets_predictions.csv`: 25,000 per-example prediction records
- `results/cifar10n_label_sets_summary.json`: experiment metadata and results
- `results/cifar10n_label_sets_metrics.png`: comparison chart
- `results/cifar10n_label_sets_hashes.txt`: reproducibility hashes


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


## Self-created dataset: AGNews-CLNoise

`AGNews-CLNoise` is a controlled noisy-label text dataset constructed from the AG News training split. It provides a cross-domain test of Confident Learning because the original paper's main quantitative benchmark used images, whereas this experiment uses news-topic text classification.

The source dataset is:

- Dataset: `fancyzhx/ag_news`
- Revision: `eb185aade064a813bc0b7f42de02595523103ca4`
- Training examples: 120,000
- Classes: World, Sports, Business and Sci/Tech

The Hugging Face dataset card currently lists the licence as unknown. Therefore, raw news text is downloaded into the ignored `data/raw/` directory and is not committed. The constructed manifest contains only source indices, labels and SHA-256 text hashes.

### Construction method

The script `scripts/agnews_noise_experiment.py`:

1. Loads the pinned AG News training split.
2. Selects a deterministic, stratified subset of 10,000 examples.
3. Retains 2,500 examples from each of the four classes.
4. Corrupts exactly 500 labels per class, producing 2,000 errors and a 20% noise rate.
5. Uses fixed cyclic transitions: World to Sports, Sports to Business, Business to Sci/Tech and Sci/Tech to World.
6. Uses four-fold cross-validation with a TF-IDF logistic-regression pipeline to produce out-of-sample probabilities.
7. Applies Cleanlab and evaluates its decisions against the known injected errors.

The cyclic transitions are an artificial reproducible stress test. They are not claimed to represent realistic human annotation behaviour.

### Run the experiment

```bash
python scripts/agnews_noise_experiment.py \
  2>&1 | tee results/agnews_noise_output.txt
```

### Results

| Measure | Result |
|---|---:|
| Constructed examples | 10,000 |
| Injected label errors | 2,000 |
| Cleanlab suspected issues | 2,251 |
| True positives | 1,701 |
| False positives | 550 |
| False negatives | 299 |
| Precision | 0.7557 |
| Recall | 0.8505 |
| F1-score | 0.8003 |
| Accuracy against noisy labels | 0.7045 |
| Accuracy against clean reference labels | 0.8559 |

Cleanlab recovered 1,701 of the 2,000 deliberately corrupted labels. The experiment demonstrates that the method can be executed on text-classification outputs, although broader conclusions require additional noise rates, transition structures, classifiers and random seeds.

The experiment was executed twice. The constructed manifest, JSON summary and all 10,000 prediction records produced identical SHA-256 hashes.

### Generated files

- `data/processed/agnews_clnoise_manifest.csv`: reproducible construction manifest without raw text
- `results/agnews_noise_output.txt`: first execution log
- `results/agnews_noise_output_run2.txt`: repeated execution log
- `results/agnews_noise_summary.json`: results and construction metadata
- `results/agnews_noise_predictions.csv`: per-example prediction results
- `results/agnews_noise_hashes_run1.txt`: first-run hashes
- `results/agnews_noise_hashes_run2.txt`: repeated-run hashes

### Controlled sensitivity study

The follow-up script `scripts/agnews_noise_sweep.py` extends the pilot into a controlled sensitivity study. It keeps the same balanced 10,000-article subset and fixed four-fold assignments while varying:

- noise rates of 10%, 20%, 30% and 40%;
- cyclic class-conditional and symmetric corruption; and
- corruption seeds 42, 43 and 44.

This produces 24 runs across eight configurations. Cleanlab is compared with a simple baseline that flags an example whenever the classifier's most probable class differs from its supplied noisy label. In addition to precision, recall and F1, the study records Matthews correlation coefficient, average precision and lift over the injected-error prevalence.

Run the study using:

```bash
python scripts/agnews_noise_sweep.py
```

Mean F1 over the three corruption seeds was:

| Noise structure | Noise rate | Cleanlab F1 | Argmax-disagreement F1 |
|---|---:|---:|---:|
| Cyclic | 10% | 0.7440 | 0.6211 |
| Cyclic | 20% | 0.8013 | 0.7349 |
| Cyclic | 30% | 0.7753 | 0.7369 |
| Cyclic | 40% | 0.6636 | 0.6611 |
| Symmetric | 10% | 0.7354 | 0.6325 |
| Symmetric | 20% | 0.8172 | 0.7733 |
| Symmetric | 30% | 0.8504 | 0.8275 |
| Symmetric | 40% | 0.8468 | 0.8480 |

Cleanlab achieved higher mean F1 in seven of the eight configurations. At 40% symmetric noise, Cleanlab and the simple baseline were effectively tied within seed-to-seed variation. The results suggest that concentrating corruption into one competing class was substantially more damaging at high noise than spreading it across three alternatives: at 40% noise, Cleanlab's mean F1 was 0.6636 under cyclic corruption and 0.8468 under symmetric corruption. This is an inference from the controlled synthetic experiment rather than a general causal claim. Precision at larger noise rates must also be interpreted alongside the increased prevalence of erroneous labels; higher precision alone does not necessarily mean that the detector has intrinsically improved.

The 20%-cyclic-seed-42 configuration exactly reproduced the original pilot counts: 2,251 flags, 1,701 true positives, 550 false positives and 299 false negatives.

### Preliminary semantic label review

Thirty Cleanlab-flagged examples that were not deliberately corrupted in the verified 20% cyclic run were subjected to a preliminary LLM-assisted semantic review. The review judged:

- 14 original labels appropriate;
- 11 original labels likely incorrect; and
- 5 examples genuinely ambiguous.

Thus, 16 of the 30 sampled apparent false positives were considered potentially questionable. These judgements are exploratory and are not treated as independent human ground truth. Raw article text remains in the ignored `data/raw/` directory; the committed review file contains source indices, SHA-256 hashes, decisions and reasons.

### Blind human--LLM agreement

The same 30 articles were subsequently assessed through an independent blind human review. For each article, the original AG News label and the classifier-supported alternative were randomly assigned to candidates A and B. The reviewer did not see which candidate represented which source and did not see the earlier LLM decision. The mapping from human and LLM responses into the categories `original`, `alternative` and `ambiguous_neither` was recorded before the human review began.

The human and LLM categories agreed on 22 of 30 cases, giving raw agreement of 73.3% and Cohen's kappa of 0.5556. The human reviewer selected the original label in 15 cases and the model-supported alternative in 15 cases. The largest disagreement involved five cases where the human selected the alternative but the LLM had marked the article ambiguous or proposed neither candidate. The positive agreement beyond chance supports using an LLM as a supplementary reviewer, but the disagreement rate shows that it should not replace human inspection.

Additional generated files are:

- `data/processed/agnews_noise_sweep_manifest.csv`
- `data/processed/agnews_manual_annotation_key.csv`
- `data/processed/agnews_manual_annotations.csv`
- `results/agnews_noise_sweep_runs.csv`
- `results/agnews_noise_sweep_aggregate.csv`
- `results/agnews_noise_sweep_summary.json`
- `results/agnews_noise_sweep_f1.png`
- `results/agnews_manual_annotation_summary.json`
- `scripts/agnews_blind_human_review.py`
- `scripts/agnews_human_llm_agreement.py`
- `results/agnews_human_llm_agreement_preregistration.json`
- `data/processed/agnews_human_llm_agreement.csv`
- `results/agnews_human_llm_agreement_matrix.csv`
- `results/agnews_human_llm_agreement_summary.json`
- `results/agnews_human_llm_agreement.png`
- `results/agnews_human_llm_agreement_hashes.txt`

### Limitations

The study uses one balanced 10,000-article subset, one TF--IDF logistic-regression classifier, two artificial corruption mechanisms and three corruption seeds. Synthetic errors do not fully represent natural annotation behaviour. The published AG News labels are used as the clean reference but may themselves contain ambiguity or mistakes. The 30-case semantic sample was selected from apparent false positives and contains only one blind human reviewer and one preliminary LLM pass. Its agreement results therefore provide exploratory evidence rather than definitive relabelling or a general estimate of LLM-judge accuracy.
