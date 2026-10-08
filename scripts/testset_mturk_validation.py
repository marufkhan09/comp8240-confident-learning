"""Reproduce published MTurk validation counts for seven test sets.

This uses the released human-validation records from:
https://github.com/cleanlab/label-errors

An example is a validated label error when fewer than three workers
select the released "given" category. Majority "both" responses form the
paper's separate multi-label error category.
"""

from collections import Counter
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


EXPECTED_SOURCE_COMMIT = (
    "6d5d6b31a13216290afc40e5c6319399c4d15c06"
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REPOSITORY = ROOT / "external/label-errors"
MTURK_DIRECTORY = SOURCE_REPOSITORY / "mturk"
RESULTS_DIRECTORY = ROOT / "results"

RESULTS_FILE = (
    RESULTS_DIRECTORY / "testset_mturk_validation.csv"
)
SUMMARY_FILE = (
    RESULTS_DIRECTORY / "testset_mturk_validation_summary.json"
)
CHART_FILE = (
    RESULTS_DIRECTORY / "testset_mturk_validation.png"
)
HASH_FILE = (
    RESULTS_DIRECTORY / "testset_mturk_validation_hashes.txt"
)

DATASETS = [
    ("mnist", "MNIST"),
    ("cifar10", "CIFAR-10"),
    ("cifar100", "CIFAR-100"),
    ("caltech256", "Caltech-256"),
    ("imagenet", "ImageNet"),
    ("20news", "20 Newsgroups"),
    ("imdb", "IMDB"),
]

# Published values from Table 2 of arXiv:2103.14749v4.
EXPECTED = {
    "mnist": {
        "candidates": 100,
        "non_errors": 85,
        "errors": 15,
        "non_agreement": 2,
        "correctable": 10,
        "multi_label": 0,
        "neither": 3,
    },
    "cifar10": {
        "candidates": 275,
        "non_errors": 221,
        "errors": 54,
        "non_agreement": 32,
        "correctable": 18,
        "multi_label": 0,
        "neither": 4,
    },
    "cifar100": {
        "candidates": 2235,
        "non_errors": 1650,
        "errors": 585,
        "non_agreement": 210,
        "correctable": 318,
        "multi_label": 20,
        "neither": 37,
    },
    "caltech256": {
        "candidates": 2360,
        "non_errors": 1902,
        "errors": 458,
        "non_agreement": 99,
        "correctable": 221,
        "multi_label": 115,
        "neither": 23,
    },
    "imagenet": {
        "candidates": 5440,
        "non_errors": 2524,
        "errors": 2916,
        "non_agreement": 598,
        "correctable": 1428,
        "multi_label": 597,
        "neither": 293,
    },
    "20news": {
        "candidates": 93,
        "non_errors": 11,
        "errors": 82,
        "non_agreement": 43,
        "correctable": 22,
        "multi_label": 12,
        "neither": 5,
    },
    "imdb": {
        "candidates": 1310,
        "non_errors": 585,
        "errors": 725,
        "non_agreement": 552,
        "correctable": 173,
        "multi_label": 0,
        "neither": 0,
    },
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(
            lambda: input_file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def classify_record(dataset_id, record):
    votes = record["mturk"]

    # Reproduce the released study's categorical decision rule.
    # The vote fields cannot be summed as a universal worker count:
    # some released MNIST records contain overlapping tallies.
    given_votes = int(votes.get("given", 0))
    guessed_votes = int(votes.get("guessed", 0))
    both_votes = int(votes.get("both", 0))
    neither_votes = int(votes.get("neither", 0))

    # The original label is retained only when at least three workers
    # selected the released "given" category.
    if given_votes >= 3:
        return "non_error"

    if dataset_id == "imdb":
        if guessed_votes >= 3:
            return "correctable"
        return "non_agreement"

    if both_votes >= 3:
        return "multi_label"

    if neither_votes >= 3:
        return "neither"

    if guessed_votes >= 3:
        return "correctable"

    return "non_agreement"


def main():
    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
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
            "Unexpected label-errors source commit: "
            f"{source_commit}"
        )

    rows = []
    source_hashes = {}

    for dataset_id, display_name in DATASETS:
        source_file = (
            MTURK_DIRECTORY
            / f"{dataset_id}_mturk.json"
        )

        records = json.loads(
            source_file.read_text(encoding="utf-8")
        )

        counts = Counter(
            classify_record(dataset_id, record)
            for record in records
        )

        actual = {
            "candidates": len(records),
            "non_errors": counts["non_error"],
            "errors": (
                len(records)
                - counts["non_error"]
            ),
            "non_agreement":
                counts["non_agreement"],
            "correctable": counts["correctable"],
            "multi_label": counts["multi_label"],
            "neither": counts["neither"],
        }

        expected = EXPECTED[dataset_id]
        exact_match = actual == expected

        if not exact_match:
            raise RuntimeError(
                f"{display_name} mismatch.\n"
                f"Actual:   {actual}\n"
                f"Expected: {expected}"
            )

        source_hashes[
            source_file.name
        ] = sha256(source_file)

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": display_name,
                **actual,
                "validated_error_percentage_of_candidates":
                    100.0
                    * actual["errors"]
                    / actual["candidates"],
                "published_counts_exact_match":
                    exact_match,
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(
        RESULTS_FILE,
        index=False,
        lineterminator="\n",
    )

    summary = {
        "experiment":
            "Released MTurk validation-count reproduction",
        "source_repository":
            "https://github.com/cleanlab/label-errors",
        "source_commit": source_commit,
        "source_paper":
            "https://arxiv.org/abs/2103.14749",
        "validation_rule":
            "Fewer than 3 workers selected the given category",
        "vote_schema_note":
            "Released category tallies are not universally additive",
        "datasets": len(results),
        "total_candidates":
            int(results["candidates"].sum()),
        "total_validated_errors":
            int(results["errors"].sum()),
        "all_published_counts_exact_match":
            bool(
                results[
                    "published_counts_exact_match"
                ].all()
            ),
        "source_file_sha256": source_hashes,
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    plt.figure(figsize=(11, 5.5))

    bars = plt.bar(
        results["dataset"],
        results[
            "validated_error_percentage_of_candidates"
        ],
        color="#2474B5",
    )

    plt.ylabel(
        "Human-validated errors among flagged candidates (%)"
    )
    plt.xlabel("Dataset")
    plt.title(
        "Human validation of Confident Learning candidates\n"
        "Released MTurk records, five workers per example"
    )
    plt.ylim(
        0,
        max(
            results[
                "validated_error_percentage_of_candidates"
            ]
        )
        * 1.16,
    )
    plt.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        results[
            "validated_error_percentage_of_candidates"
        ],
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(
        CHART_FILE,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()

    hash_targets = [
        Path(__file__).resolve(),
        RESULTS_FILE,
        SUMMARY_FILE,
        CHART_FILE,
    ]

    HASH_FILE.write_text(
        "".join(
            f"{sha256(path)}  "
            f"{path.relative_to(ROOT)}\n"
            for path in hash_targets
        ),
        encoding="utf-8",
    )

    print(
        "TEST-SET MTURK VALIDATION REPRODUCTION"
    )
    print("Source commit:", source_commit)
    print(
        "Rule: fewer than 3 workers selected "
        "the given-label category"
    )
    print()

    print(
        f"{'Dataset':15s} "
        f"{'Candidates':>10s} "
        f"{'Errors':>8s} "
        f"{'Confirmed':>10s} "
        f"{'Match':>7s}"
    )

    for row in rows:
        print(
            f"{row['dataset']:15s} "
            f"{row['candidates']:10d} "
            f"{row['errors']:8d} "
            f"{row['validated_error_percentage_of_candidates']:9.1f}% "
            f"{'YES':>7s}"
        )

    print()
    print(
        "Total candidates:",
        summary["total_candidates"],
    )
    print(
        "Total validated errors:",
        summary["total_validated_errors"],
    )
    print(
        "ALL SEVEN DATASETS: EXACT MATCH"
    )
    print()
    print("Results:", RESULTS_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)


if __name__ == "__main__":
    main()
