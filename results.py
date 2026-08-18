"""
Analyze results from the sandbagging.py eval run.

Reads all .eval log files in a log directory (default: ./logs) and produces:

  1. A comparison table of accuracy for each task under the "regular" vs
     "sandbagging" condition, with the delta between them (mirrors the
     paper's Table 1 / Fig. 4).

  2. A refusal / invalid-answer rate per task, based on how often the
     `choice()` scorer could not parse a valid A/B/C/D answer out of the
     model's response (mirrors the paper's Appendix D refusal analysis).

Usage:
    python analyze_results.py                # uses ./logs
    python analyze_results.py --logs my_logs  # custom log dir
    python analyze_results.py --csv out.csv   # also save table to CSV
"""

import argparse
import re

import pandas as pd
from inspect_ai.analysis import SampleScores, evals_df, samples_df

VALID_LETTERS = {"A", "B", "C", "D"}


def condition_from_task_name(task_name: str) -> tuple[str, str]:
    """Split e.g. 'wmdp_bio_sandbagging' -> ('wmdp_bio', 'sandbagging')."""
    if task_name.endswith("_sandbagging"):
        return task_name[: -len("_sandbagging")], "sandbagging"
    if task_name.endswith("_regular"):
        return task_name[: -len("_regular")], "regular"
    return task_name, "unknown"


def build_accuracy_table(logs_dir: str) -> pd.DataFrame:
    """Part 1: regular vs sandbagging accuracy per task, with delta."""
    df = evals_df(logs=logs_dir)

    if df.empty:
        raise SystemExit(f"No eval logs found in '{logs_dir}'.")

    df[["benchmark", "condition"]] = df["task_name"].apply(
        lambda t: pd.Series(condition_from_task_name(t))
    )

    # score_headline_value holds the primary metric (accuracy, for choice()).
    pivot = df.pivot_table(
        index="benchmark",
        columns="condition",
        values="score_headline_value",
        aggfunc="first",
    )

    if "regular" in pivot.columns and "sandbagging" in pivot.columns:
        pivot["delta"] = pivot["sandbagging"] - pivot["regular"]
        pivot["delta_pct"] = (pivot["delta"] * 100).round(1)

    return pivot.sort_index()


def find_score_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Locate the choice() scorer's answer/value columns regardless of exact naming.

    Inspect's score_details() extractor names the bare correct/incorrect value
    column "score_<scorer>" (e.g. "score_choice") and the parsed answer column
    "score_<scorer>_answer" (e.g. "score_choice_answer") -- there's no "_value"
    suffix on the base column, so we match on the "_answer" suffix and infer
    the corresponding value column by stripping it.
    """
    answer_cols = [c for c in df.columns if re.match(r"score_.*_answer$", c)]
    if not answer_cols:
        raise SystemExit(
            f"Could not find a score '_answer' column. Available columns: {list(df.columns)}"
        )
    answer_col = answer_cols[0]
    value_col = answer_col[: -len("_answer")]
    if value_col not in df.columns:
        raise SystemExit(
            f"Found answer column '{answer_col}' but matching value column "
            f"'{value_col}' is missing. Available columns: {list(df.columns)}"
        )
    return answer_col, value_col


def build_refusal_table(logs_dir: str, task_filter: str = "wmdp") -> pd.DataFrame:
    """Part 3: refusal / invalid-answer rate per task.

    A sample counts as a "refusal/invalid" if the choice() scorer's parsed
    `answer` field is empty or not one of A/B/C/D -- i.e. the model didn't
    give a clean lettered answer for the multiple_choice() solver to parse.
    """
    # The default samples_df() columns only include the bare correct/incorrect
    # score value. SampleScores adds the *_answer / *_explanation detail
    # fields we need to detect refusals (requires full=True to load).
    df = samples_df(logs=logs_dir, columns=SampleScores, full=True)
    if df.empty:
        raise SystemExit(f"No sample data found in '{logs_dir}'.")

    if "task_name" not in df.columns:
        # samples_df doesn't always include task_name directly; merge it in
        # from evals_df via the eval_id / log path if needed.
        evals = evals_df(logs=logs_dir)[["eval_id", "task_name"]]
        df = df.merge(evals, on="eval_id", how="left")

    if task_filter:
        df = df[df["task_name"].str.contains(task_filter)]

    answer_col, _ = find_score_columns(df)

    def is_invalid(answer) -> bool:
        if answer is None or (isinstance(answer, float) and pd.isna(answer)):
            return True
        letters = {c.upper() for c in str(answer) if c.isalpha()}
        return not letters.issubset(VALID_LETTERS) or len(letters) == 0

    df["invalid"] = df[answer_col].apply(is_invalid)
    df[["benchmark", "condition"]] = df["task_name"].apply(
        lambda t: pd.Series(condition_from_task_name(t))
    )

    summary = (
        df.groupby(["benchmark", "condition"])["invalid"]
        .agg(invalid_count="sum", total="count")
        .assign(invalid_rate_pct=lambda d: (d["invalid_count"] / d["total"] * 100).round(1))
    )

    return summary.sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs", help="Path to the Inspect log directory")
    parser.add_argument("--csv", default=None, help="Optional path to save accuracy table as CSV")
    args = parser.parse_args()

    print("=" * 70)
    print("ACCURACY: regular vs sandbagging (paper Table 1 / Fig. 4 style)")
    print("=" * 70)
    acc_table = build_accuracy_table(args.logs)
    print(acc_table.to_string())

    if args.csv:
        acc_table.to_csv(args.csv)
        print(f"\nSaved to {args.csv}")

    print()
    print("=" * 70)
    print("REFUSAL / INVALID-ANSWER RATE (paper Appendix D style)")
    print("=" * 70)
    refusal_table = build_refusal_table(args.logs, task_filter="wmdp")
    print(refusal_table.to_string())


if __name__ == "__main__":
    main()