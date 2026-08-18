"""
plot_eval_results.py

Reads pairs of Inspect AI `.eval` log files (a "regular" run and a
"sandbagging" run per task) and produces a grouped, stacked bar chart
showing, per run:

    - Correct                (solid)
    - Refused / null answer  (hatched slashes)
    - Incorrect              (solid gray)

Regular runs are colored orange; sandbagging runs are colored a darker
burnt-orange/red, so the two conditions are distinguishable even when
several tasks are plotted side by side. Each task gets ONE x-axis label
(centered under its pair of bars) instead of one label per file, so labels
don't overlap.

"Refused / null answer" is inferred as any sample whose choice-scorer
`answer` field is empty, i.e. the model never emitted a parsable
"ANSWER: <letter>" in its completion (as opposed to answering with the
wrong letter, which counts as a normal incorrect).

Requirements:
    pip install inspect_ai matplotlib

Usage:
    python plot_eval_results.py -o output.png

Edit the TASKS list below to point at your own .eval files. Add as many
tasks as you like; each renders as one Regular/Sandbagging bar pair.
"""

import argparse

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

# ---------------------------------------------------------------------------
# Configure your tasks here. Each entry needs a display name plus the
# regular and sandbagging .eval file paths for that task.
# ---------------------------------------------------------------------------
TASKS = [
    {
        "name": "WMDP-Bio",
        "group": "Target",
        "regular_path": "logs/2026-08-05T12-08-14-00-00_wmdp-bio-regular_ebC7U5YorAe3bfR9p6cN79.eval",
        "sandbagging_path": "logs/2026-08-05T12-09-19-00-00_wmdp-bio-sandbagging_jZKsaaUar6jfgcsh4LSmwf.eval",
        "random_baseline": 0.25,
    },
     {
         "name": "WMDP-Chem",
         "group": "Target",
         "regular_path": 'logs/2026-08-05T11-59-51-00-00_wmdp-chem-regular_MrYy7J9ok5YFy7Lnmc6Jao.eval',
         "sandbagging_path": 'logs/2026-08-05T12-00-09-00-00_wmdp-chem-sandbagging_LD8okV8GKWiTNnZw793FUE.eval',
         "random_baseline": 0.25,
     },
     {
         "name": "WMDP-Cyber",
         "group": "Target",
         "regular_path": 'logs/2026-08-05T12-00-22-00-00_wmdp-cyber-regular_Mj33bEFpnMT6aHdhko55W2.eval',
         "sandbagging_path": 'logs/2026-08-05T12-01-10-00-00_wmdp-cyber-sandbagging_DaywxD6e6jujnvesEQYNRh.eval',
         "random_baseline": 0.25,
     },
]

# Regular = orange family, Sandbagging = darker burnt-orange/red family
REGULAR_CORRECT = "#E8952A"
REGULAR_INCORRECT_GRAY = "#8C8C8C"

SANDBAGGING_CORRECT = "#C1440E"
SANDBAGGING_INCORRECT_GRAY = "#8C8C8C"


def score_log(path):
    """Return (correct, refused, incorrect, total) for one .eval file."""
    log = read_eval_log(path)

    correct = 0
    refused = 0
    incorrect = 0

    for sample in log.samples:
        sc = sample.scores["choice"]
        if sc.value == "C":
            correct += 1
        elif not sc.answer:  # no parsable answer -> refusal / null response
            refused += 1
        else:  # a real (wrong) answer was given
            incorrect += 1

    total = correct + refused + incorrect
    return correct, refused, incorrect, total


def add_stack(ax, xpos, width, correct, refused, incorrect, total,
              correct_color, gray_color, label_prefix, show_legend):
    """Draw one stacked bar (correct / refused-hatched / incorrect) at xpos."""
    b_correct = ax.bar(xpos, correct, width=width, color=correct_color,
                        edgecolor="black", linewidth=0.8,
                        label=f"{label_prefix}: Correct" if show_legend else None)
    b_refused = ax.bar(xpos, refused, width=width, bottom=correct, color=gray_color,
                        edgecolor='black', linewidth=0.8, hatch="//",
                        label=f"{label_prefix}: Refused/Null" if show_legend else None)
    b_incorrect = ax.bar(xpos, incorrect, width=width, bottom=correct + refused,
                          color=gray_color, edgecolor="black", linewidth=0.8,
                          label=f"{label_prefix}: Incorrect" if show_legend else None)

    # Annotate segments (skip tiny/zero ones to avoid clutter)
    def annotate(val, bottom, color):
        if val == 0:
            return
        ax.text(xpos, bottom + val / 2, str(val), ha="center", va="center",
                 fontsize=9, color="white" if val > total * 0.04 else "black")

    annotate(correct, 0, correct_color)
    annotate(refused, correct, gray_color)
    annotate(incorrect, correct + refused, gray_color)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="eval_results.png",
                         help="Path to save the resulting PNG chart")
    args = parser.parse_args()

    bar_width = 0.35
    gap_between_tasks = 1.0
    x_positions = [i * gap_between_tasks for i in range(len(TASKS))]

    fig, ax = plt.subplots(figsize=(3 + 2.5 * len(TASKS), 6))

    first = True
    for xpos, task in zip(x_positions, TASKS):
        r_correct, r_refused, r_incorrect, r_total = score_log(task["regular_path"])
        s_correct, s_refused, s_incorrect, s_total = score_log(task["sandbagging_path"])

        print(f"{task['name']} (Regular): total={r_total} correct={r_correct} "
              f"refused/null={r_refused} incorrect={r_incorrect}")
        print(f"{task['name']} (Sandbagging): total={s_total} correct={s_correct} "
              f"refused/null={s_refused} incorrect={s_incorrect}")

        add_stack(ax, xpos - bar_width / 2, bar_width, r_correct, r_refused, r_incorrect,
                   r_total, REGULAR_CORRECT, REGULAR_INCORRECT_GRAY, "Regular", first)
        add_stack(ax, xpos + bar_width / 2, bar_width, s_correct, s_refused, s_incorrect,
                   s_total, SANDBAGGING_CORRECT, SANDBAGGING_INCORRECT_GRAY, "Sandbagging", first)
        first = False

    # One x-tick label per task, centered between its two bars
    ax.set_xticks(x_positions)
    ax.set_xticklabels([t["name"] for t in TASKS])
    ax.set_ylabel("Number of samples")
    ax.set_title("Eval Results: Correct vs. Refused/Null vs. Incorrect")

    # Simplify the legend: one swatch per condition/segment combo, but
    # reorganize into two rows (Regular row, Sandbagging row) for clarity.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart to {args.output}")


if __name__ == "__main__":
    main()