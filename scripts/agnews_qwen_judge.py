"""Formal, order-controlled LLM-judge pilot for AG News.

A pinned Qwen2.5-0.5B-Instruct model judges 30 disputed examples twice.
Candidate order is deterministically randomised in pass 1 and reversed in
pass 2. Raw article text and model responses remain under data/raw/.
"""

from pathlib import Path
import hashlib
import json
import os
import re
import time

import cleanlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = (
    "7ae557604adf67be50417f59c2c2f167def9a775"
)

ORDER_SEED = 20261008
GENERATION_SEED = 42
BATCH_SIZE = 4
MAX_INPUT_TOKENS = 768
MAX_NEW_TOKENS = 64

CATEGORIES = [
    "original",
    "alternative",
    "ambiguous_neither",
]

DISPLAY_NAMES = [
    "Original",
    "Alternative",
    "Ambiguous/neither",
]

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT
    / "data/raw/agnews_blind_human_review_completed.csv"
)

PRIVATE_OUTPUT = (
    ROOT
    / "data/raw/agnews_qwen_judge_private.csv"
)

MODEL_CACHE = (
    ROOT
    / "data/raw/huggingface/qwen2_5_0_5b_instruct"
)

PUBLIC_OUTPUT = (
    ROOT
    / "data/processed/agnews_qwen_judge_results.csv"
)

PROMPT_FILE = (
    ROOT
    / "results/agnews_qwen_judge_prompt.txt"
)

PROTOCOL_FILE = (
    ROOT
    / "results/agnews_qwen_judge_protocol.json"
)

SUMMARY_FILE = (
    ROOT
    / "results/agnews_qwen_judge_summary.json"
)

MATRIX_FILE = (
    ROOT
    / "results/agnews_qwen_judge_matrix.csv"
)

CHART_FILE = (
    ROOT
    / "results/agnews_qwen_judge_agreement.png"
)

HASH_FILE = (
    ROOT
    / "results/agnews_qwen_judge_hashes.txt"
)


SYSTEM_PROMPT = """You are a careful evaluator of news-topic labels.

The four possible AG News topics are:
- World: international affairs, politics, governments, conflict or society.
- Sports: athletes, teams, competitions or sporting organisations.
- Business: companies, markets, finance, employment or commercial activity.
- Sci/Tech: science, computing, technology, engineering or space.

For each article, compare Candidate A and Candidate B.

Choose:
- A if Candidate A is clearly the better topic;
- B if Candidate B is clearly the better topic;
- AMBIGUOUS if both candidates are reasonably defensible;
- NEITHER if neither candidate is appropriate.

Judge only from the supplied article. Do not assume either candidate is the
published label or a model prediction.

Return exactly two lines:
CHOICE: A, B, AMBIGUOUS, or NEITHER
REASON: one brief sentence
"""


PROMPT_TEMPLATE = """ARTICLE:
{article}

CANDIDATE A: {candidate_a}
CANDIDATE B: {candidate_b}

Return exactly:
CHOICE: <A, B, AMBIGUOUS, or NEITHER>
REASON: <one brief sentence>
"""


def sha256_text(value):
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def write_hash_file(paths):
    lines = []

    for path in paths:
        digest = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        relative = path.relative_to(ROOT)
        lines.append(f"{digest}  {relative}")

    HASH_FILE.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def parse_choice(output):
    normalised = output.strip().upper()

    patterns = [
        r"CHOICE\s*:\s*(AMBIGUOUS|NEITHER|A|B)\b",
        r"^\s*(AMBIGUOUS|NEITHER|A|B)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalised)

        if match:
            return match.group(1)

    return "INVALID"


def choice_to_semantic(choice, candidate_a_role, candidate_b_role):
    if choice == "A":
        return candidate_a_role

    if choice == "B":
        return candidate_b_role

    if choice == "AMBIGUOUS":
        return "ambiguous"

    if choice == "NEITHER":
        return "neither"

    return "invalid"


def collapse_semantic(value):
    if value in {"original", "alternative"}:
        return value

    if value in {"ambiguous", "neither"}:
        return "ambiguous_neither"

    return "invalid"


def agreement_metrics(frame, prediction_column):
    valid = frame[
        frame[prediction_column].isin(CATEGORIES)
    ].copy()

    if valid.empty:
        return {
            "evaluated_cases": 0,
            "agreements": 0,
            "raw_agreement": None,
            "cohens_kappa": None,
        }

    agreements = int(
        np.sum(
            valid[prediction_column].to_numpy()
            == valid["human_category"].to_numpy()
        )
    )

    raw_agreement = agreements / len(valid)

    kappa = cohen_kappa_score(
        valid["human_category"],
        valid[prediction_column],
        labels=CATEGORIES,
    )

    return {
        "evaluated_cases": int(len(valid)),
        "agreements": agreements,
        "raw_agreement": float(raw_agreement),
        "cohens_kappa": (
            float(kappa)
            if not np.isnan(kappa)
            else None
        ),
    }


def build_records(input_frame):
    rng = np.random.default_rng(ORDER_SEED)
    records = []

    for row in input_frame.itertuples(index=False):
        original = str(row.verified_original_label)
        alternative = str(row.model_alternative_label)

        if original == alternative:
            raise RuntimeError(
                f"Identical candidates for review ID {row.review_id}."
            )

        swap = bool(rng.integers(0, 2))

        if swap:
            first_a = alternative
            first_b = original
            first_a_role = "alternative"
            first_b_role = "original"
        else:
            first_a = original
            first_b = alternative
            first_a_role = "original"
            first_b_role = "alternative"

        orders = [
            (
                1,
                first_a,
                first_b,
                first_a_role,
                first_b_role,
            ),
            (
                2,
                first_b,
                first_a,
                first_b_role,
                first_a_role,
            ),
        ]

        for (
            pass_number,
            candidate_a,
            candidate_b,
            candidate_a_role,
            candidate_b_role,
        ) in orders:
            user_prompt = PROMPT_TEMPLATE.format(
                article=str(row.article_text),
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )

            records.append(
                {
                    "review_id": int(row.review_id),
                    "source_index": int(row.source_index),
                    "article_text": str(row.article_text),
                    "text_sha256": sha256_text(
                        str(row.article_text)
                    ),
                    "human_category": str(
                        row.reviewer_agreement_category
                    ),
                    "verified_original_label": original,
                    "model_alternative_label": alternative,
                    "pass_number": pass_number,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "candidate_a_role": candidate_a_role,
                    "candidate_b_role": candidate_b_role,
                    "user_prompt": user_prompt,
                }
            )

    return records


def generate_responses(records):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    torch.manual_seed(GENERATION_SEED)
    np.random.seed(GENERATION_SEED)

    available_threads = os.cpu_count() or 1
    torch.set_num_threads(
        max(1, min(4, available_threads))
    )

    print("Loading pinned tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE),
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading pinned model on CPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(MODEL_CACHE),
        dtype=torch.float32,
    )

    model.to("cpu")
    model.eval()

    # Greedy decoding is deliberately deterministic. Clear sampling-only
    # values inherited from the model's generation configuration.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    responses = []

    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[
            start:start + BATCH_SIZE
        ]

        conversations = []

        for record in batch_records:
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": record["user_prompt"],
                },
            ]

            conversations.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        encoded = tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        input_width = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for row_number, sequence in enumerate(generated):
            generated_tokens = sequence[input_width:]

            output = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            record = dict(batch_records[row_number])
            record["raw_model_output"] = output
            record["choice"] = parse_choice(output)

            record["semantic_category"] = (
                choice_to_semantic(
                    record["choice"],
                    record["candidate_a_role"],
                    record["candidate_b_role"],
                )
            )

            record["comparison_category"] = (
                collapse_semantic(
                    record["semantic_category"]
                )
            )

            responses.append(record)

        completed = min(
            start + BATCH_SIZE,
            len(records),
        )

        print(
            f"  Completed {completed}/{len(records)} prompts"
        )

    del model

    return responses


def create_case_results(response_frame):
    rows = []

    for review_id, group in response_frame.groupby(
        "review_id",
        sort=True,
    ):
        if len(group) != 2:
            raise RuntimeError(
                f"Review ID {review_id} does not have two passes."
            )

        first = group[
            group["pass_number"] == 1
        ].iloc[0]

        second = group[
            group["pass_number"] == 2
        ].iloc[0]

        first_valid = (
            first["comparison_category"] in CATEGORIES
        )

        second_valid = (
            second["comparison_category"] in CATEGORIES
        )

        exact_consistent = (
            first_valid
            and second_valid
            and first["semantic_category"]
            == second["semantic_category"]
        )

        collapsed_consistent = (
            first_valid
            and second_valid
            and first["comparison_category"]
            == second["comparison_category"]
        )

        rows.append(
            {
                "review_id": int(review_id),
                "source_index": int(first["source_index"]),
                "text_sha256": first["text_sha256"],
                "verified_original_label":
                    first["verified_original_label"],
                "model_alternative_label":
                    first["model_alternative_label"],
                "human_category":
                    first["human_category"],
                "pass1_candidate_a":
                    first["candidate_a"],
                "pass1_candidate_b":
                    first["candidate_b"],
                "pass1_candidate_a_role":
                    first["candidate_a_role"],
                "pass1_candidate_b_role":
                    first["candidate_b_role"],
                "pass1_choice":
                    first["choice"],
                "pass1_semantic_category":
                    first["semantic_category"],
                "pass1_comparison_category":
                    first["comparison_category"],
                "pass1_valid": bool(first_valid),
                "pass2_candidate_a":
                    second["candidate_a"],
                "pass2_candidate_b":
                    second["candidate_b"],
                "pass2_candidate_a_role":
                    second["candidate_a_role"],
                "pass2_candidate_b_role":
                    second["candidate_b_role"],
                "pass2_choice":
                    second["choice"],
                "pass2_semantic_category":
                    second["semantic_category"],
                "pass2_comparison_category":
                    second["comparison_category"],
                "pass2_valid": bool(second_valid),
                "exact_order_consistent":
                    bool(exact_consistent),
                "collapsed_order_consistent":
                    bool(collapsed_consistent),
            }
        )

    return pd.DataFrame(rows)


def create_chart(case_results, summary):
    consistent = case_results[
        case_results["collapsed_order_consistent"]
    ].copy()

    if consistent.empty:
        matrix = np.zeros((3, 3), dtype=int)
    else:
        matrix = confusion_matrix(
            consistent["human_category"],
            consistent["pass1_comparison_category"],
            labels=CATEGORIES,
        )

    matrix_frame = pd.DataFrame(
        matrix,
        index=[
            f"Human: {name}"
            for name in DISPLAY_NAMES
        ],
        columns=[
            f"Judge: {name}"
            for name in DISPLAY_NAMES
        ],
    )

    matrix_frame.to_csv(
        MATRIX_FILE,
        lineterminator="\n",
    )

    metric_names = [
        "Pass 1\nvalid",
        "Pass 2\nvalid",
        "Order\nconsistent",
        "Pass 1\nhuman agreement",
        "Pass 2\nhuman agreement",
        "Consensus\nhuman agreement",
    ]

    metric_values = [
        summary["pass1_valid_rate"],
        summary["pass2_valid_rate"],
        summary["collapsed_order_consistency_rate"],
        summary["pass1_human_agreement"]["raw_agreement"]
            or 0.0,
        summary["pass2_human_agreement"]["raw_agreement"]
            or 0.0,
        summary["consistent_consensus_human_agreement"][
            "raw_agreement"
        ] or 0.0,
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    colours = [
        "#4C78A8",
        "#4C78A8",
        "#F58518",
        "#54A24B",
        "#54A24B",
        "#B279A2",
    ]

    bars = axes[0].bar(
        metric_names,
        metric_values,
        color=colours,
    )

    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Proportion")
    axes[0].set_title(
        "Formal Qwen judge: validity, consistency and agreement"
    )
    axes[0].grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, metric_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    image = axes[1].imshow(
        matrix,
        cmap="Blues",
    )

    axes[1].set_xticks(
        range(3),
        DISPLAY_NAMES,
        rotation=25,
        ha="right",
    )

    axes[1].set_yticks(
        range(3),
        DISPLAY_NAMES,
    )

    axes[1].set_xlabel("Order-consistent Qwen judgement")
    axes[1].set_ylabel("Blind human judgement")
    axes[1].set_title(
        "Agreement matrix for order-consistent cases"
    )

    for row in range(3):
        for column in range(3):
            axes[1].text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, column]
                    > matrix.max() / 2
                    and matrix.max() > 0
                    else "black"
                ),
                fontweight="bold",
            )

    figure.colorbar(
        image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )

    figure.savefig(
        CHART_FILE,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def main():
    started = time.perf_counter()

    for directory in [
        PRIVATE_OUTPUT.parent,
        PUBLIC_OUTPUT.parent,
        SUMMARY_FILE.parent,
        MODEL_CACHE,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    input_frame = pd.read_csv(INPUT_FILE)

    required_columns = {
        "review_id",
        "source_index",
        "article_text",
        "verified_original_label",
        "model_alternative_label",
        "reviewer_agreement_category",
    }

    missing = required_columns.difference(
        input_frame.columns
    )

    if missing:
        raise RuntimeError(
            f"Missing input columns: {sorted(missing)}"
        )

    if len(input_frame) != 30:
        raise RuntimeError(
            "Expected exactly 30 reviewed cases."
        )

    invalid_human = set(
        input_frame[
            "reviewer_agreement_category"
        ]
    ).difference(CATEGORIES)

    if invalid_human:
        raise RuntimeError(
            f"Unexpected human categories: {invalid_human}"
        )

    prompt_document = (
        SYSTEM_PROMPT
        + "\nUSER-PROMPT TEMPLATE\n"
        + PROMPT_TEMPLATE
    )

    PROMPT_FILE.write_text(
        prompt_document,
        encoding="utf-8",
    )

    protocol = {
        "experiment":
            "Formal order-controlled AG News LLM-judge pilot",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_role":
            "Supplementary semantic judge, not ground truth",
        "cases": 30,
        "passes_per_case": 2,
        "total_prompts": 60,
        "order_seed": ORDER_SEED,
        "generation_seed": GENERATION_SEED,
        "pass_1_order":
            "Deterministically randomised candidate order",
        "pass_2_order":
            "Exact reversal of pass 1 candidate order",
        "decoding":
            "Greedy decoding; sampling disabled; one beam",
        "allowed_choices": [
            "A",
            "B",
            "AMBIGUOUS",
            "NEITHER",
        ],
        "comparison_mapping": {
            "A": "Role assigned to candidate A",
            "B": "Role assigned to candidate B",
            "AMBIGUOUS": "ambiguous_neither",
            "NEITHER": "ambiguous_neither",
            "unparseable": "invalid",
        },
        "human_categories": CATEGORIES,
        "privacy":
            "Raw text and raw model responses remain in data/raw",
        "python_version":
            ".".join(
                str(value)
                for value in os.sys.version_info[:3]
            ),
        "pytorch_version": torch.__version__,
        "transformers_version":
            transformers.__version__,
        "scikit_learn_version":
            sklearn.__version__,
        "cleanlab_version":
            cleanlab.__version__,
    }

    PROTOCOL_FILE.write_text(
        json.dumps(
            protocol,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("FORMAL AG NEWS LLM-JUDGE PILOT")
    print("Model:", MODEL_ID)
    print("Pinned revision:", MODEL_REVISION)
    print("Cases:", len(input_frame))
    print("Prompts:", len(input_frame) * 2)
    print("Device: CPU")
    print()

    records = build_records(input_frame)
    responses = generate_responses(records)

    response_frame = pd.DataFrame(responses)

    response_frame.to_csv(
        PRIVATE_OUTPUT,
        index=False,
        lineterminator="\n",
    )

    case_results = create_case_results(
        response_frame
    )

    case_results.to_csv(
        PUBLIC_OUTPUT,
        index=False,
        lineterminator="\n",
    )

    pass1_valid = int(
        case_results["pass1_valid"].sum()
    )

    pass2_valid = int(
        case_results["pass2_valid"].sum()
    )

    both_valid = (
        case_results["pass1_valid"]
        & case_results["pass2_valid"]
    )

    both_valid_count = int(both_valid.sum())

    exact_consistent_count = int(
        case_results[
            "exact_order_consistent"
        ].sum()
    )

    collapsed_consistent_count = int(
        case_results[
            "collapsed_order_consistent"
        ].sum()
    )

    pass1_agreement = agreement_metrics(
        case_results,
        "pass1_comparison_category",
    )

    pass2_agreement = agreement_metrics(
        case_results,
        "pass2_comparison_category",
    )

    consistent = case_results[
        case_results[
            "collapsed_order_consistent"
        ]
    ].copy()

    consensus_agreement = agreement_metrics(
        consistent,
        "pass1_comparison_category",
    )

    summary = {
        "experiment":
            "Formal order-controlled AG News LLM-judge pilot",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "cases": int(len(case_results)),
        "total_prompts": int(len(response_frame)),
        "pass1_valid_outputs": pass1_valid,
        "pass2_valid_outputs": pass2_valid,
        "pass1_valid_rate":
            pass1_valid / len(case_results),
        "pass2_valid_rate":
            pass2_valid / len(case_results),
        "both_passes_valid": both_valid_count,
        "exact_order_consistent_cases":
            exact_consistent_count,
        "exact_order_consistency_rate": (
            exact_consistent_count / both_valid_count
            if both_valid_count
            else 0.0
        ),
        "collapsed_order_consistent_cases":
            collapsed_consistent_count,
        "collapsed_order_consistency_rate": (
            collapsed_consistent_count / both_valid_count
            if both_valid_count
            else 0.0
        ),
        "pass1_human_agreement":
            pass1_agreement,
        "pass2_human_agreement":
            pass2_agreement,
        "consistent_consensus_human_agreement":
            consensus_agreement,
        "limitations": [
            "Small 30-case sample",
            "Small 0.5B parameter judge",
            "CPU-only greedy decoding",
            "Two candidate labels rather than open classification",
            "Human review is one reviewer, not expert consensus",
        ],
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    create_chart(
        case_results,
        summary,
    )

    write_hash_file(
        [
            Path(__file__).resolve(),
            PROMPT_FILE,
            PROTOCOL_FILE,
            PUBLIC_OUTPUT,
            MATRIX_FILE,
            SUMMARY_FILE,
            CHART_FILE,
        ]
    )

    elapsed = time.perf_counter() - started

    print()
    print("FORMAL LLM-JUDGE RESULTS")
    print(
        f"Valid outputs, pass 1: "
        f"{pass1_valid}/30 "
        f"({summary['pass1_valid_rate']:.1%})"
    )
    print(
        f"Valid outputs, pass 2: "
        f"{pass2_valid}/30 "
        f"({summary['pass2_valid_rate']:.1%})"
    )
    print(
        f"Exact order consistency: "
        f"{exact_consistent_count}/{both_valid_count} "
        f"({summary['exact_order_consistency_rate']:.1%})"
    )
    print(
        f"Collapsed order consistency: "
        f"{collapsed_consistent_count}/{both_valid_count} "
        f"({summary['collapsed_order_consistency_rate']:.1%})"
    )

    for label, values in [
        ("Pass 1 versus human", pass1_agreement),
        ("Pass 2 versus human", pass2_agreement),
        (
            "Order-consistent consensus versus human",
            consensus_agreement,
        ),
    ]:
        agreement = values["raw_agreement"]
        kappa = values["cohens_kappa"]

        agreement_text = (
            f"{agreement:.1%}"
            if agreement is not None
            else "N/A"
        )

        kappa_text = (
            f"{kappa:.4f}"
            if kappa is not None
            else "N/A"
        )

        print(
            f"{label}: "
            f"n={values['evaluated_cases']} "
            f"agreement={agreement_text} "
            f"kappa={kappa_text}"
        )

    print()
    print(
        "3x3 consensus matrix "
        "(rows=human, columns=judge):"
    )
    print(pd.read_csv(MATRIX_FILE, index_col=0))

    print()
    print(f"Total runtime: {elapsed:.2f} seconds")
    print("Public results:", PUBLIC_OUTPUT)
    print("Summary:", SUMMARY_FILE)
    print("Matrix:", MATRIX_FILE)
    print("Chart:", CHART_FILE)
    print("Hashes:", HASH_FILE)
    print("Private raw-text output:", PRIVATE_OUTPUT)


if __name__ == "__main__":
    main()
