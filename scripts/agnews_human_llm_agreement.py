"""Compare the blind AG News human review with the earlier LLM review."""

from collections import Counter
from pathlib import Path
import hashlib
import json

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


CATEGORIES = ["original", "alternative", "ambiguous_neither"]
DISPLAY_NAMES = ["Original", "Alternative", "Ambiguous/neither"]

ROOT = Path(__file__).resolve().parents[1]

PRIVATE_INPUT = (
    ROOT / "data/raw/agnews_blind_human_review_completed.csv"
)
PREREGISTRATION = (
    ROOT / "results/agnews_human_llm_agreement_preregistration.json"
)
PUBLIC_COMPARISON = (
    ROOT / "data/processed/agnews_human_llm_agreement.csv"
)
MATRIX_FILE = (
    ROOT / "results/agnews_human_llm_agreement_matrix.csv"
)
SUMMARY_FILE = (
    ROOT / "results/agnews_human_llm_agreement_summary.json"
)
CHART_FILE = (
    ROOT / "results/agnews_human_llm_agreement.png"
)
HASH_FILE = (
    ROOT / "results/agnews_human_llm_agreement_hashes.txt"
)


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def map_llm_category(row):
    decision = clean_text(row["llm_decision"]).lower()
    suggestion = clean_text(row["llm_suggested_label"])
    alternative = clean_text(row["model_alternative_label"])

    if decision == "correct":
        return "original"

    if decision == "ambiguous":
        return "ambiguous_neither"

    if decision == "incorrect":
        if suggestion == alternative:
            return "alternative"

        # A suggested third label follows the preregistered rule.
        return "ambiguous_neither"

    raise ValueError(
        f"Unexpected LLM decision: {decision!r}"
    )


def text_hash(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def write_hash_file(paths):
    lines = []

    for path in paths:
        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        relative_path = path.relative_to(ROOT)
        lines.append(
            f"{digest}  {relative_path}"
        )

    HASH_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    if not PRIVATE_INPUT.exists():
        raise FileNotFoundError(
            "Completed blind-review file was not found."
        )

    if not PREREGISTRATION.exists():
        raise FileNotFoundError(
            "Preregistration file was not found."
        )

    review = pd.read_csv(
        PRIVATE_INPUT,
        keep_default_na=False,
    )

    required_columns = {
        "review_id",
        "source_index",
        "article_text",
        "verified_original_label",
        "model_alternative_label",
        "reviewer_choice",
        "reviewer_selected_label",
        "reviewer_agreement_category",
        "llm_decision",
        "llm_suggested_label",
    }

    missing_columns = (
        required_columns - set(review.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Missing columns: "
            + str(sorted(missing_columns))
        )

    if len(review) != 30:
        raise RuntimeError(
            f"Expected 30 cases, found {len(review)}."
        )

    if review["reviewer_choice"].eq("").any():
        raise RuntimeError(
            "At least one human decision is blank."
        )

    human_categories = review[
        "reviewer_agreement_category"
    ].astype(str)

    invalid_categories = sorted(
        set(human_categories) - set(CATEGORIES)
    )

    if invalid_categories:
        raise RuntimeError(
            "Unexpected human categories: "
            + str(invalid_categories)
        )

    llm_categories = review.apply(
        map_llm_category,
        axis=1,
    )

    agreements = (
        human_categories == llm_categories
    )

    agreement_count = int(agreements.sum())
    disagreement_count = int(
        len(review) - agreement_count
    )
    raw_agreement = float(agreements.mean())

    kappa = float(
        cohen_kappa_score(
            human_categories,
            llm_categories,
            labels=CATEGORIES,
        )
    )

    matrix = confusion_matrix(
        human_categories,
        llm_categories,
        labels=CATEGORIES,
    )

    matrix_frame = pd.DataFrame(
        matrix,
        index=DISPLAY_NAMES,
        columns=DISPLAY_NAMES,
    )
    matrix_frame.index.name = "Human category"

    # This public file contains hashes, labels and decisions,
    # but never contains raw article text.
    public_comparison = pd.DataFrame(
        {
            "review_id":
                review["review_id"].astype(int),
            "source_index":
                review["source_index"].astype(int),
            "text_sha256":
                review["article_text"].map(text_hash),
            "original_label":
                review["verified_original_label"],
            "model_alternative_label":
                review["model_alternative_label"],
            "human_choice":
                review["reviewer_choice"],
            "human_selected_label":
                review["reviewer_selected_label"],
            "human_category":
                human_categories,
            "llm_decision":
                review["llm_decision"],
            "llm_suggested_label":
                review["llm_suggested_label"],
            "llm_category":
                llm_categories,
            "agreement":
                agreements,
        }
    ).sort_values("review_id")

    summary = {
        "study":
            "AG News blind human versus LLM agreement",
        "cases": int(len(review)),
        "category_order": CATEGORIES,
        "human_category_counts":
            dict(Counter(human_categories)),
        "llm_category_counts":
            dict(Counter(llm_categories)),
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "raw_agreement": raw_agreement,
        "cohens_kappa": kappa,
        "mapping_source":
            str(PREREGISTRATION.relative_to(ROOT)),
        "privacy":
            "No raw article text is stored in public outputs.",
    }

    PUBLIC_COMPARISON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    MATRIX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    public_comparison.to_csv(
        PUBLIC_COMPARISON,
        index=False,
    )

    matrix_frame.to_csv(MATRIX_FILE)

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(
        figsize=(7.2, 5.6)
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
    )

    maximum = matrix.max()

    for row_index in range(3):
        for column_index in range(3):
            value = int(
                matrix[row_index, column_index]
            )

            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
                color=(
                    "white"
                    if value > maximum / 2
                    else "#152238"
                ),
            )

    axis.set_xticks(
        range(3),
        DISPLAY_NAMES,
    )
    axis.set_yticks(
        range(3),
        DISPLAY_NAMES,
    )

    axis.set_xlabel("Earlier LLM review")
    axis.set_ylabel("Blind human review")

    axis.set_title(
        f"Human–LLM agreement: "
        f"{agreement_count}/30 "
        f"({raw_agreement:.1%}), "
        f"κ={kappa:.3f}"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Number of cases",
    )

    figure.tight_layout()

    figure.savefig(
        CHART_FILE,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

    write_hash_file(
        [
            ROOT / "scripts/agnews_blind_human_review.py",
            Path(__file__).resolve(),
            PREREGISTRATION,
            PUBLIC_COMPARISON,
            MATRIX_FILE,
            SUMMARY_FILE,
            CHART_FILE,
        ]
    )

    print(
        "HUMAN-LLM AGREEMENT ANALYSIS COMPLETE"
    )
    print(f"Cases: {len(review)}")
    print(f"Agreements: {agreement_count}")
    print(
        f"Disagreements: {disagreement_count}"
    )
    print(
        f"Raw agreement: "
        f"{raw_agreement:.4f} "
        f"({raw_agreement:.1%})"
    )
    print(f"Cohen's kappa: {kappa:.4f}")

    print()
    print(
        "3x3 agreement table "
        "(rows=human, columns=LLM):"
    )
    print(matrix_frame.to_string())

    print()
    print(
        "Public comparison:",
        PUBLIC_COMPARISON,
    )
    print("Summary:", SUMMARY_FILE)
    print("Matrix:", MATRIX_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)
    print("Raw article text committed: NO")


if __name__ == "__main__":
    main()
