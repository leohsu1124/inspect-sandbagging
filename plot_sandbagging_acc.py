"""
plot_sandbagging_accuracy.py

Reproduces the "Regular vs. Sandbagging" grouped bar chart style (as seen in
sandbagging papers, e.g. the Claude 3 evals chart) from Inspect AI `.eval`
log files.

For each task, accuracy is computed as:

    correct / (correct + incorrect)

i.e. samples with a null / unparsable answer are EXCLUDED from both the
numerator and denominator entirely (they don't count as wrong, they just
don't count).

Tasks are grouped into "Control" (capability sanity checks, e.g. CSQA/MMLU)
and "Target" (the benchmark(s) being sandbagged, e.g. WMDP-Bio/Chem/Cyber).
A dotted "Random baseline" line marks chance accuracy (e.g. 25% for 4-choice
MCQ).

Requirements:
    pip install inspect_ai matplotlib

Usage:
    python plot_sandbagging_accuracy.py -o output.png

Edit the TASKS list below to point at your own .eval files. Any task whose
regular_path or sandbagging_path is None is skipped (bar omitted), so you
can fill in only the tasks you actually have logs for.
"""

import argparse

import matplotlib.pyplot as plt
from inspect_ai.log import read_eval_log

# ---------------------------------------------------------------------------
# Configure your tasks here. Each entry needs:
#   name             - label shown under the bar group
#   group            - "Control" or "Target" (or any grouping label you like)
#   regular_path     - path to the .eval log for the "regular" (honest) run
#   sandbagging_path - path to the .eval log for the "sandbagging" run
#   random_baseline  - chance accuracy as a fraction, e.g. 0.25 for 4-choice
# Set a path to None to skip that half of a task's bars.
# ---------------------------------------------------------------------------
TASKS = [
    # --- Control tasks (fill in with your own CSQA / MMLU logs) ---
     {
         "name": "CSQA",
         "group": "Control",
         "regular_path": 'logs/2026-07-28T03-55-17-00-00_csqa-regular_GB7fMwxKNZUUEPnhSE9ZqP.eval',
         "sandbagging_path": 'logs/2026-07-28T03-55-52-00-00_csqa-sandbagging_2CiYHxyi4TK2opL4ek38qX.eval',
         "random_baseline": 0.20,
     },
     {
         "name": "MMLU",
         "group": "Control",
         "regular_path": 'logs/2026-07-28T03-38-46-00-00_mmlu-regular_XS6BHkWaayvJkohr7YCmE4.eval',
         "sandbagging_path": 'logs/2026-07-28T03-47-04-00-00_mmlu-sandbagging_K8EggcRxU6EsQ6VP5bQVoY.eval',
         "random_baseline": 0.25,
     },
    # --- Target tasks ---
    {
        "name": "WMDP-Bio",
        "group": "Target",
        "regular_path": "logs/2026-07-28T02-45-21-00-00_wmdp-bio-regular_VofzSEjvXZ4rnnfr24ptxR.eval",
        "sandbagging_path": "logs/2026-07-28T02-46-03-00-00_wmdp-bio-sandbagging_EXhgr39nwyLiT39MK28Bid.eval",
        "random_baseline": 0.25,
    },
     {
         "name": "WMDP-Chem",
         "group": "Target",
         "regular_path": 'logs/2026-07-28T02-46-37-00-00_wmdp-chem-regular_LgwFWtU64M2n2wZ4fHQnx4.eval',
         "sandbagging_path": 'logs/2026-07-28T02-47-58-00-00_wmdp-chem-sandbagging_diX22dxzcsN3xqoX6QAKKw.eval',
         "random_baseline": 0.25,
     },
     {
         "name": "WMDP-Cyber",
         "group": "Target",
         "regular_path": 'logs/2026-08-05T10-00-39-00-00_wmdp-cyber-regular_Kvx5sSypRTGRGGjyGPpBHv.eval',
         "sandbagging_path": 'logs/2026-08-05T10-03-46-00-00_wmdp-cyber-sandbagging_4DPiDnZo7VEbbBPWF5bMG2.eval',
         "random_baseline": 0.25,
     },
]

REGULAR_COLOR = "#E45858"
SANDBAGGING_COLOR = "#A51D1D"

def accuracy(path):
    """Return accuracy (%) = correct / (correct + incorrect), ignoring null answers."""
    if path is None:
        return None

    log = read_eval_log(path)

    correct = 0
    incorrect = 0

    for sample in log.samples:
        sc = sample.scores["choice"]
        if sc.value == "C":
            correct += 1
        elif sc.answer:  # non-empty answer that was wrong
            incorrect += 1
        # else: empty/null answer -> excluded entirely

    answered = correct + incorrect
    if answered == 0:
        return None
    return 100.0 * correct / answered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default="sandbagging_accuracy.png",
                         help="Path to save the resulting PNG chart")
    parser.add_argument("--title", default="Claude Sonnet 5 Sandbagging Accuracy",
                         help="Chart title")
    args = parser.parse_args()

    # Compute accuracies
    for task in TASKS:
        task["regular_acc"] = accuracy(task["regular_path"])
        task["sandbagging_acc"] = accuracy(task["sandbagging_path"])
        print(f"{task['name']}: regular={task['regular_acc']}  sandbagging={task['sandbagging_acc']}")

    # Drop tasks with no data at all
    tasks = [t for t in TASKS if t["regular_acc"] is not None or t["sandbagging_acc"] is not None]
    if not tasks:
        raise SystemExit("No tasks with data found — check TASKS paths.")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_width = 0.35
    gap_between_tasks = 1.0
    x_positions = [i * gap_between_tasks for i in range(len(tasks))]

    reg_bars = []
    sand_bars = []

    for xpos, task in zip(x_positions, tasks):
        if task["regular_acc"] is not None:
            b = ax.bar(xpos - bar_width / 2, task["regular_acc"], width=bar_width,
                       color=REGULAR_COLOR, edgecolor="black", linewidth=0.6,
                       label="Regular" if not reg_bars else None)
            reg_bars.append(b)
            ax.text(xpos - bar_width / 2, task["regular_acc"] + 1.5, f"{task['regular_acc']:.0f}%",
                    ha="center", va="bottom", color=REGULAR_COLOR, fontweight="bold")

        if task["sandbagging_acc"] is not None:
            b = ax.bar(xpos + bar_width / 2, task["sandbagging_acc"], width=bar_width,
                       color=SANDBAGGING_COLOR, edgecolor="black", linewidth=0.6,
                       label="Sandbagging" if not sand_bars else None)
            sand_bars.append(b)
            ax.text(xpos + bar_width / 2, task["sandbagging_acc"] + 1.5, f"{task['sandbagging_acc']:.0f}%",
                    ha="center", va="bottom", color=SANDBAGGING_COLOR, fontweight="bold")

    # Random baseline dotted line(s) — draw one segment per task in case
    # baselines differ, but if they're all equal it reads as one line.
    baseline_label_used = False
    for xpos, task in zip(x_positions, tasks):
        baseline = task.get("random_baseline")
        if baseline is None:
            continue
        y = baseline * 100
        ax.plot([xpos - 0.5, xpos + 0.5], [y, y], linestyle=":", color="gray",
                linewidth=1.3, label="Random baseline" if not baseline_label_used else None)
        baseline_label_used = True

    # X tick labels (task names)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([t["name"] for t in tasks])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title(args.title)

    # Group brackets/labels below the task names (e.g. "Control" / "Target")
    groups = []
    for xpos, task in zip(x_positions, tasks):
        if groups and groups[-1][0] == task["group"]:
            groups[-1][2] = xpos
        else:
            groups.append([task["group"], xpos, xpos])

    trans = ax.get_xaxis_transform()  # x in data coords, y in axes coords
    label_y = -0.16
    line_y = -0.10
    for group_name, x_start, x_end in groups:
        ax.plot([x_start - bar_width, x_end + bar_width], [line_y, line_y],
                color="black", linewidth=1, clip_on=False, transform=trans)
        ax.text((x_start + x_end) / 2, label_y, group_name, ha="center", va="top",
                transform=trans, fontsize=11)

    # Legend
    ax.legend(loc="lower left", frameon=True)

    # Clean up spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved chart to {args.output}")


if __name__ == "__main__":
    main()