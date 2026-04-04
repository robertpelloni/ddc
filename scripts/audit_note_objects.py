import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date

SYMBOL_TO_SEMANTIC = {
    "0": "EMPTY",
    "1": "TAP",
    "2": "HOLD_HEAD",
    "3": "TAIL",
    "4": "ROLL_HEAD",
    "A": "ATTACK",
    "F": "FAKE",
    "K": "KEYSOUND",
    "L": "LIFT",
    "M": "MINE",
}


def audit_json_dir(json_dir: str):
    global_counts = Counter()
    per_bucket_counts = defaultdict(Counter)
    charts_with_symbol = defaultdict(Counter)
    row_examples = defaultdict(Counter)
    chart_count = 0

    for filename in sorted(os.listdir(json_dir)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(json_dir, filename), "r", encoding="utf-8") as f:
            meta = json.load(f)
        for chart in meta.get("charts", []):
            chart_count += 1
            bucket = f"{chart.get('type', 'unknown')}_{chart.get('difficulty_coarse', 'unknown')}"
            present = set()
            for _, _, _, note in chart.get("notes", []):
                chars = set(note)
                for symbol in chars:
                    if symbol in SYMBOL_TO_SEMANTIC and symbol != "0":
                        row_examples[symbol][note] += 1
                for symbol in note:
                    if symbol in SYMBOL_TO_SEMANTIC and symbol != "0":
                        global_counts[symbol] += 1
                        per_bucket_counts[bucket][symbol] += 1
                        present.add(symbol)
            for symbol in present:
                charts_with_symbol[bucket][symbol] += 1

    return chart_count, global_counts, per_bucket_counts, charts_with_symbol, row_examples


def render_markdown(json_dir: str, chart_count: int, global_counts, per_bucket_counts, charts_with_symbol, row_examples):
    today = date.today().isoformat()
    lines = [
        "# Note Object Semantics Audit",
        "",
        f"Date: {today}",
        "",
        "## Scope",
        "",
        f"- Extracted JSON directory: `{json_dir}`",
        f"- Charts audited: **{chart_count}**",
        "",
        "## Symbol Mapping",
        "",
        "| Symbol | Interpreted semantic | Present in refreshed corpus? | Raw count |",
        "|---|---|---:|---:|",
    ]

    for symbol, semantic in SYMBOL_TO_SEMANTIC.items():
        if symbol == "0":
            continue
        count = global_counts.get(symbol, 0)
        lines.append(f"| `{symbol}` | `{semantic}` | {'Yes' if count else 'No'} | {count} |")

    lines += [
        "",
        "## Key Findings",
        "",
        "- `1` (tap) dominates the corpus, as expected.",
        "- `2` and `3` are both present in substantial quantities, strongly indicating hold-head/tail usage in the official DDR corpus.",
        "- `M` is present and corresponds to mines.",
        "- `4`, `A`, `F`, `K`, and `L` were not observed in the refreshed extracted corpus audit.",
        "- This means the current extracted corpus clearly contains taps, hold-related structure, and mines, but no observed roll-head, attack, fake, keysound, or lift symbols in the audited official-pack extraction.",
        "",
        "## Per-Bucket Counts",
        "",
        "| Bucket | Chart count with symbol | Symbol totals |",
        "|---|---|---|",
    ]

    for bucket in sorted(per_bucket_counts):
        chart_summary = ", ".join(
            f"`{symbol}` charts={charts_with_symbol[bucket].get(symbol, 0)}"
            for symbol in sorted(per_bucket_counts[bucket])
        )
        count_summary = ", ".join(
            f"`{symbol}`={per_bucket_counts[bucket][symbol]}"
            for symbol in sorted(per_bucket_counts[bucket])
        )
        lines.append(f"| `{bucket}` | {chart_summary} | {count_summary} |")

    lines += ["", "## Example Rows by Symbol", ""]
    for symbol in ["2", "3", "M"]:
        lines.append(f"### `{symbol}` / `{SYMBOL_TO_SEMANTIC[symbol]}`")
        lines.append("")
        if row_examples[symbol]:
            for row, count in row_examples[symbol].most_common(15):
                lines.append(f"- `{row}`: {count}")
        else:
            lines.append("- None observed")
        lines.append("")

    lines += [
        "## Practical Implications",
        "",
        "- The DDC symbolic training corpus is not tap-only; it includes hold-heads, tails, and mines.",
        "- The difficulty evaluator still currently reduces charts to tap notes only, so these semantics are not yet fully represented there.",
        "- The next useful improvement is to extend difficulty-feature extraction to account for hold and mine burden explicitly.",
    ]

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit note-object semantics in extracted JSON charts.")
    parser.add_argument("--json-dir", default="data/training_output/json_raw/ddr_official")
    parser.add_argument("--out", default="docs/NOTE_OBJECT_SEMANTICS_2026-04-04.md")
    args = parser.parse_args()

    chart_count, global_counts, per_bucket_counts, charts_with_symbol, row_examples = audit_json_dir(args.json_dir)
    markdown = render_markdown(args.json_dir, chart_count, global_counts, per_bucket_counts, charts_with_symbol, row_examples)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)


if __name__ == "__main__":
    main()
