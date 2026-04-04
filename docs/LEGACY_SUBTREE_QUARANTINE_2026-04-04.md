# Legacy Subtree Quarantine Decision

Date: 2026-04-04

## Purpose

This note records the practical decision to treat the remaining unresolved merge-conflict-marker files inside the `ddc_stepmania/` subtree as **legacy/quarantined content** rather than immediate blockers for the active modernization path.

---

## Remaining files with conflict markers

At the time of this note, the repository-health audit reports that the only remaining unresolved merge-conflict-marker files are:

- `ddc_stepmania/learn/beatcalc.py`
- `ddc_stepmania/learn/chart.py`

These are contained entirely within the `ddc_stepmania` subtree.

---

## Why quarantine instead of immediate cleanup

## 1. They are outside the main active execution path

The current active modernization / integration work has focused on the main repository path, including:

- `dataset/`
- `learn/`
- `scripts/`
- `infer/`
- `ffr-difficulty-model/`
- root docs / packaging / orchestration

The `ddc_stepmania/` subtree is legacy-oriented and is not the main path used for the current PyTorch-oriented training and audit work.

## 2. Risk-to-value ratio is lower than the active retraining path

The highest-value remaining execution work is now:

- full `.ssc`-inclusive retraining refresh
- refreshed difficulty-evaluator training
- deployment/export preparation

Cleaning the final two conflict-marker files in a legacy subtree may still be worth doing later, but it is no longer the most leverage-heavy next action.

## 3. Main-path repository health is now substantially improved

Successive cleanup passes reduced unresolved merge-conflict-marker files from:

- **15 -> 9 -> 4 -> 2**

So the remaining problem surface is already confined to legacy content rather than the main operational path.

---

## Practical interpretation

The repo is not literally conflict-marker-free yet.

However, the unresolved markers are now:

- isolated
- legacy-scoped
- non-central to the current active training/inference modernization work

That means the repository can reasonably proceed with the `.ssc`-inclusive retraining refresh while tracking this legacy cleanup as a secondary maintenance task.

---

## Recommended policy

### Short-term

Treat `ddc_stepmania/` conflict-marker files as **quarantined legacy subtree content**.

### Medium-term

If the `ddc_stepmania/` subtree is expected to remain maintained:

- normalize those two files in a dedicated cleanup pass
- then rerun the repository health audit

If the subtree is effectively archival:

- leave it quarantined and clearly documented as legacy content
- keep active development focused on the main modernized path

---

## Recommended next action after this decision

Proceed with:

1. full `.ssc`-inclusive filtering/feature refresh
2. refreshed DDC training
3. refreshed difficulty-evaluator training
4. deploy/export packaging

while keeping the final two legacy-subtree conflict-marker files on the maintenance backlog.
