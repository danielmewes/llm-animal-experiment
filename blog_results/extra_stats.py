"""Builds blog_run/extra_stats.csv from the per-configuration JSON results."""
import csv
import glob
import json
import os

from llm_animal_experiment.experiment import ANIMALS
from llm_animal_experiment.stats import run_statistical_test

here = os.path.dirname(os.path.abspath(__file__))


def picked_animal(trial):
    """The animal the model's answers reveal as its pick (None if unresolved/INVALID)."""
    if trial["classification"] == "YES":
        return trial["guessed_animal"]
    for fu in trial["follow_ups"]:
        if fu["classification"] == "YES":
            return fu["guessed_animal"]
    return None


rows = []
for path in sorted(glob.glob(os.path.join(here, "*.json"))):
    d = json.load(open(path))
    name = os.path.basename(path)[:-5]
    model, level, variant = name.rsplit("_", 2)
    trials = d["trials"]
    yes = sum(t["classification"] == "YES" for t in trials)
    no = sum(t["classification"] == "NO" for t in trials)
    invalid = len(trials) - yes - no
    valid = yes + no
    tests = {
        c: run_statistical_test(yes, no, invalid, 1 / 3, c) for c in (0.95, 0.99)
    }
    picks = [picked_animal(t) for t in trials]
    resolved = [p for p in picks if p]
    row = {
        "model": model, "thinking_level": level, "prompt": variant,
        "total_trials": len(trials), "valid_trials": valid, "invalid_trials": invalid,
        "yes_count": yes, "no_count": no,
        "yes_pct": round(100 * yes / valid, 1) if valid else "",
        "no_pct": round(100 * no / valid, 1) if valid else "",
    }
    for c, label in ((0.95, "p95"), (0.99, "p99")):
        row[f"{label}_p_value"] = round(tests[c].binom_pvalue, 6)
        row[f"{label}_null_hypothesis"] = "REJECTED" if tests[c].reject_null else "not rejected"
    row["animal_trials_resolved"] = len(resolved)
    for a in ANIMALS:
        row[f"{a}_pct"] = round(100 * resolved.count(a) / len(resolved), 1) if resolved else ""
    rows.append(row)

out = os.path.join(here, "extra_stats.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out}")
