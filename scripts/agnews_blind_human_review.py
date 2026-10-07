"""Conduct a blind human review of 30 AG News cases.

The reviewer sees:
- article text
- two randomly ordered candidate labels, A and B

The reviewer does not see:
- which candidate is the original AG News label
- which candidate is the classifier prediction
- the earlier LLM decision
- the earlier LLM explanation

Progress is stored only under data/raw/, which is ignored by Git.
"""

from pathlib import Path
from collections import Counter
import random
import textwrap

import numpy as np
import pandas as pd


RANDOM_SEED = 8240

CLASS_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

PRIVATE_INPUT = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "agnews_manual_annotation_private.csv"
)

PREDICTIONS_INPUT = (
    REPOSITORY_ROOT
    / "results"
    / "agnews_noise_predictions.csv"
)

PROGRESS_FILE = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "agnews_blind_human_review_progress.csv"
)

COMPLETED_FILE = (
    REPOSITORY_ROOT
    / "data"
    / "raw"
    / "agnews_blind_human_review_completed.csv"
)


def clean_optional_value(value):
    if pd.isna(value):
        return ""
    return str(value)


def prepare_review_sheet():
    private = pd.read_csv(PRIVATE_INPUT)
    predictions = pd.read_csv(PREDICTIONS_INPUT)

    required_private = {
        "review_id",
        "source_index",
        "article_text",
        "original_agnews_label",
        "human_decision",
        "reason",
        "suggested_label",
    }

    required_predictions = {
        "source_index",
        "clean_label",
        "predicted_label",
    }

    missing_private = required_private - set(private.columns)
    missing_predictions = required_predictions - set(
        predictions.columns
    )

    if missing_private:
        raise RuntimeError(
            f"Private review file is missing: {missing_private}"
        )

    if missing_predictions:
        raise RuntimeError(
            f"Prediction file is missing: {missing_predictions}"
        )

    prediction_subset = predictions[
        ["source_index", "clean_label", "predicted_label"]
    ].copy()

    review = private.merge(
        prediction_subset,
        on="source_index",
        how="left",
        validate="one_to_one",
    )

    if review[["clean_label", "predicted_label"]].isna().any().any():
        raise RuntimeError(
            "Some review cases could not be matched to predictions."
        )

    review["clean_label"] = review["clean_label"].astype(int)
    review["predicted_label"] = review["predicted_label"].astype(int)

    review["verified_original_label"] = review["clean_label"].map(
        CLASS_NAMES
    )
    review["model_alternative_label"] = review[
        "predicted_label"
    ].map(CLASS_NAMES)

    mismatched_originals = review[
        review["original_agnews_label"]
        != review["verified_original_label"]
    ]

    if len(mismatched_originals):
        raise RuntimeError(
            "The private original labels do not match clean_label."
        )

    identical_candidates = review[
        review["verified_original_label"]
        == review["model_alternative_label"]
    ]

    if len(identical_candidates):
        raise RuntimeError(
            "At least one case has identical candidate labels."
        )

    # Preserve the earlier review as LLM evidence. These values are never
    # displayed during the blind human review.
    review["llm_decision"] = review["human_decision"].apply(
        clean_optional_value
    )
    review["llm_reason"] = review["reason"].apply(
        clean_optional_value
    )
    review["llm_suggested_label"] = review[
        "suggested_label"
    ].apply(clean_optional_value)

    review_ids = review["review_id"].astype(int).tolist()

    # Exactly half the cases place the original label in position A.
    assignment_ids = review_ids.copy()
    assignment_rng = random.Random(RANDOM_SEED)
    assignment_rng.shuffle(assignment_ids)
    original_in_a = set(
        assignment_ids[: len(assignment_ids) // 2]
    )

    # Review cases are also presented in a deterministic random order.
    presentation_ids = review_ids.copy()
    order_rng = random.Random(RANDOM_SEED + 1)
    order_rng.shuffle(presentation_ids)

    presentation_order = {
        review_id: order_number
        for order_number, review_id in enumerate(
            presentation_ids,
            start=1,
        )
    }

    candidate_a = []
    candidate_b = []
    candidate_a_role = []
    candidate_b_role = []

    for row in review.itertuples(index=False):
        original = row.verified_original_label
        alternative = row.model_alternative_label

        if int(row.review_id) in original_in_a:
            candidate_a.append(original)
            candidate_b.append(alternative)
            candidate_a_role.append("original")
            candidate_b_role.append("alternative")
        else:
            candidate_a.append(alternative)
            candidate_b.append(original)
            candidate_a_role.append("alternative")
            candidate_b_role.append("original")

    review["candidate_a"] = candidate_a
    review["candidate_b"] = candidate_b
    review["candidate_a_role"] = candidate_a_role
    review["candidate_b_role"] = candidate_b_role

    review["presentation_order"] = review["review_id"].map(
        presentation_order
    )

    review["reviewer_choice"] = ""
    review["reviewer_selected_label"] = ""
    review["reviewer_agreement_category"] = ""
    review["reviewer_reason"] = ""

    columns = [
        "presentation_order",
        "review_id",
        "source_index",
        "article_text",
        "candidate_a",
        "candidate_b",
        "candidate_a_role",
        "candidate_b_role",
        "verified_original_label",
        "model_alternative_label",
        "reviewer_choice",
        "reviewer_selected_label",
        "reviewer_agreement_category",
        "reviewer_reason",
        "llm_decision",
        "llm_suggested_label",
        "llm_reason",
    ]

    review = review[columns].sort_values(
        "presentation_order"
    ).reset_index(drop=True)

    return review


def save_progress(review):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(PROGRESS_FILE, index=False)


def choose_third_label(candidate_a, candidate_b):
    remaining = [
        label
        for label in CLASS_NAMES.values()
        if label not in {candidate_a, candidate_b}
    ]

    print()
    print("Neither candidate was selected.")
    print("Choose the better third label:")

    for number, label in enumerate(remaining, start=1):
        print(f"  {number} = {label}")

    while True:
        selection = input(
            f"Third label [1-{len(remaining)}]: "
        ).strip()

        if selection.isdigit():
            index = int(selection) - 1
            if 0 <= index < len(remaining):
                return remaining[index]

        print("Please enter a valid number.")


def conduct_review(review):
    unanswered = review[
        review["reviewer_choice"].fillna("").eq("")
    ]

    if len(unanswered) == 0:
        return review

    total = len(review)

    print()
    print("=" * 78)
    print("BLIND AG NEWS HUMAN REVIEW")
    print("=" * 78)
    print("A and B are presented in random order.")
    print("You will not be shown which label came from which source.")
    print()
    print("Choices:")
    print("  a = Candidate A")
    print("  b = Candidate B")
    print("  n = Neither candidate")
    print("  u = Ambiguous")
    print("  q = Save and quit")
    print()
    print("Your answer is saved after every case.")

    for index, row in review.iterrows():
        existing = clean_optional_value(
            row["reviewer_choice"]
        )

        if existing:
            continue

        completed = int(
            review["reviewer_choice"]
            .fillna("")
            .ne("")
            .sum()
        )

        print()
        print("=" * 78)
        print(
            f"Case {completed + 1}/{total} "
            f"(blind review number "
            f"{int(row['presentation_order'])})"
        )
        print("-" * 78)

        wrapped_text = textwrap.fill(
            str(row["article_text"]),
            width=78,
        )
        print(wrapped_text)
        print("-" * 78)
        print(f"Candidate A: {row['candidate_a']}")
        print(f"Candidate B: {row['candidate_b']}")
        print("-" * 78)

        while True:
            choice = input(
                "Decision [a/b/n/u/q]: "
            ).strip().lower()

            if choice in {"a", "b", "n", "u", "q"}:
                break

            print(
                "Enter a, b, n, u, or q."
            )

        if choice == "q":
            save_progress(review)
            print()
            print(
                f"Progress saved: {completed}/{total} completed."
            )
            print(f"File: {PROGRESS_FILE}")
            return review

        if choice == "a":
            selected_label = row["candidate_a"]
            agreement_category = row["candidate_a_role"]
            stored_choice = "A"

        elif choice == "b":
            selected_label = row["candidate_b"]
            agreement_category = row["candidate_b_role"]
            stored_choice = "B"

        elif choice == "n":
            selected_label = choose_third_label(
                row["candidate_a"],
                row["candidate_b"],
            )
            agreement_category = "ambiguous_neither"
            stored_choice = "Neither"

        else:
            selected_label = ""
            agreement_category = "ambiguous_neither"
            stored_choice = "Ambiguous"

        reason = input(
            "Brief reason (optional; press Enter to skip): "
        ).strip()

        review.at[index, "reviewer_choice"] = stored_choice
        review.at[index, "reviewer_selected_label"] = (
            selected_label
        )
        review.at[
            index,
            "reviewer_agreement_category",
        ] = agreement_category
        review.at[index, "reviewer_reason"] = reason

        save_progress(review)

    return review


def finish_review(review):
    unanswered = review[
        review["reviewer_choice"].fillna("").eq("")
    ]

    if len(unanswered):
        return

    review.to_csv(COMPLETED_FILE, index=False)

    counts = Counter(
        review["reviewer_choice"].tolist()
    )

    category_counts = Counter(
        review["reviewer_agreement_category"].tolist()
    )

    print()
    print("=" * 78)
    print("BLIND HUMAN REVIEW COMPLETE")
    print("=" * 78)
    print(f"Cases reviewed: {len(review)}")
    print(f"Candidate A selected: {counts.get('A', 0)}")
    print(f"Candidate B selected: {counts.get('B', 0)}")
    print(f"Neither selected: {counts.get('Neither', 0)}")
    print(f"Ambiguous: {counts.get('Ambiguous', 0)}")
    print()
    print("Decoded categories:")
    print(
        "Original label selected:",
        category_counts.get("original", 0),
    )
    print(
        "Model alternative selected:",
        category_counts.get("alternative", 0),
    )
    print(
        "Ambiguous/neither:",
        category_counts.get("ambiguous_neither", 0),
    )
    print()
    print(f"Completed private file: {COMPLETED_FILE}")
    print("The earlier LLM decisions remained hidden during review.")


def main():
    if PROGRESS_FILE.exists():
        review = pd.read_csv(
            PROGRESS_FILE,
            keep_default_na=False,
        )
        completed = int(
            review["reviewer_choice"].ne("").sum()
        )
        print(
            f"Resuming saved review: "
            f"{completed}/{len(review)} completed."
        )
    else:
        review = prepare_review_sheet()
        save_progress(review)
        print(
            "Created a new deterministic blind-review sheet."
        )

    review = conduct_review(review)
    finish_review(review)


if __name__ == "__main__":
    main()
