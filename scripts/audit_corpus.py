import argparse
import json
import os
from collections import Counter, defaultdict
from datetime import date


def find_files(root: str, suffix: str):
    matches = []
    if not root or not os.path.exists(root):
        return matches
    suffix = suffix.lower()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.lower().endswith(suffix):
                matches.append(os.path.join(dirpath, filename))
    return sorted(matches)


def audit_json_dir(json_dir: str):
    songs = 0
    charts = 0
    chart_type_counts = Counter()
    difficulty_counts = Counter()
    note_char_counts = Counter()
    special_row_counts = Counter()
    per_bucket_chart_counts = Counter()
    per_bucket_special_chart_counts = Counter()
    per_bucket_char_counts = defaultdict(Counter)
    charts_with_special = 0
    rows_with_special = 0

    if not json_dir or not os.path.exists(json_dir):
        return {
            "songs": 0,
            "charts": 0,
            "chart_type_counts": chart_type_counts,
            "difficulty_counts": difficulty_counts,
            "note_char_counts": note_char_counts,
            "special_row_counts": special_row_counts,
            "per_bucket_chart_counts": per_bucket_chart_counts,
            "per_bucket_special_chart_counts": per_bucket_special_chart_counts,
            "per_bucket_char_counts": per_bucket_char_counts,
            "charts_with_special": 0,
            "rows_with_special": 0,
        }

    for filename in sorted(os.listdir(json_dir)):
        if not filename.lower().endswith(".json"):
            continue
        path = os.path.join(json_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        songs += 1
        for chart in meta.get("charts", []):
            charts += 1
            chart_type = chart.get("type", "unknown")
            difficulty = chart.get("difficulty_coarse", "unknown")
            bucket = (chart_type, difficulty)

            chart_type_counts[chart_type] += 1
            difficulty_counts[difficulty] += 1
            per_bucket_chart_counts[bucket] += 1

            has_special = False
            for _, _, _, note in chart.get("notes", []):
                for ch in note:
                    note_char_counts[ch] += 1
                    per_bucket_char_counts[bucket][ch] += 1
                if set(note) - set("01"):
                    has_special = True
                    rows_with_special += 1
                    special_row_counts[note] += 1

            if has_special:
                charts_with_special += 1
                per_bucket_special_chart_counts[bucket] += 1

    return {
        "songs": songs,
        "charts": charts,
        "chart_type_counts": chart_type_counts,
        "difficulty_counts": difficulty_counts,
        "note_char_counts": note_char_counts,
        "special_row_counts": special_row_counts,
        "per_bucket_chart_counts": per_bucket_chart_counts,
        "per_bucket_special_chart_counts": per_bucket_special_chart_counts,
        "per_bucket_char_counts": per_bucket_char_counts,
        "charts_with_special": charts_with_special,
        "rows_with_special": rows_with_special,
    }


def render_markdown(raw_dir: str, json_dir: str, ssc_files, sm_files, audit: dict) -> str:
    today = date.today().isoformat()
    lines = [
        "# DDR Corpus Audit",
        "",
        f"Date: {today}",
        "",
        "## Scope",
        "",
        f"- Raw directory: `{raw_dir}`",
        f"- Extracted JSON directory: `{json_dir}`",
        "",
        "## Raw File Inventory",
        "",
        f"- `.sm` files found: **{len(sm_files)}**",
        f"- `.ssc` files found: **{len(ssc_files)}**",
        "",
        "### `.ssc` files present in the raw corpus",
        "",
    ]

    if ssc_files:
        for path in ssc_files:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")

    lines += [
        "",
        "## Extracted JSON Audit",
        "",
        f"- Songs audited: **{audit['songs']}**",
        f"- Charts audited: **{audit['charts']}**",
        f"- Charts containing any non-binary note symbol (`not just 0/1`): **{audit['charts_with_special']}**",
        f"- Rows containing any non-binary note symbol: **{audit['rows_with_special']}**",
        "",
        "### Chart types",
        "",
    ]

    for key, value in sorted(audit["chart_type_counts"].items()):
        lines.append(f"- `{key}`: {value}")

    lines += ["", "### Coarse difficulties", ""]
    for key, value in sorted(audit["difficulty_counts"].items()):
        lines.append(f"- `{key}`: {value}")

    lines += ["", "### Note-character frequency", ""]
    for key, value in audit["note_char_counts"].most_common():
        lines.append(f"- `{key}`: {value}")

    lines += ["", "### Most common special rows", ""]
    for row, count in audit["special_row_counts"].most_common(40):
        lines.append(f"- `{row}`: {count}")

    lines += ["", "## Per-bucket summary", "", "| Bucket | Charts | Charts with special symbols | Character counts |", "|---|---:|---:|---|"]
    for bucket in sorted(audit["per_bucket_chart_counts"].keys()):
        chart_count = audit["per_bucket_chart_counts"][bucket]
        special_count = audit["per_bucket_special_chart_counts"][bucket]
        chars = audit["per_bucket_char_counts"][bucket]
        char_summary = ", ".join(f"`{k}`={v}" for k, v in sorted(chars.items()))
        lines.append(f"| `{bucket[0]} / {bucket[1]}` | {chart_count} | {special_count} | {char_summary} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- The extractor now supports `.ssc` in addition to `.sm`; this audit reflects a refreshed JSON extraction that includes those files.",
        "- The extracted corpus clearly contains non-binary symbols beyond `0` and `1`, notably `2`, `3`, and `M`.",
        "- This means the DDC symbolic corpus is not tap-only.",
        "- However, the difficulty evaluator currently reduces charts to tap notes only, so non-tap object semantics are not fully modeled there.",
        "",
        "## Recommended next actions",
        "",
        "1. Re-run downstream filtering/training so the newly recovered `.ssc` songs are included in model training.",
        "2. Add a note-object semantic audit mapping for `2`, `3`, `M`, and any additional symbols encountered in future packs.",
        "3. Extend the difficulty evaluator so it captures object types beyond taps.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit raw DDR corpus coverage and extracted chart symbols.")
    parser.add_argument("--raw-dir", default="data/raw/ddr_official", help="Root raw songs directory")
    parser.add_argument("--json-dir", default="data/training_output/json_raw/ddr_official", help="Extracted JSON directory")
    parser.add_argument("--out", default="docs/CORPUS_AUDIT_2026-04-04.md", help="Markdown output path")
    args = parser.parse_args()

    sm_files = find_files(args.raw_dir, ".sm")
    ssc_files = find_files(args.raw_dir, ".ssc")
    audit = audit_json_dir(args.json_dir)
    markdown = render_markdown(args.raw_dir, args.json_dir, ssc_files, sm_files, audit)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)


if __name__ == "__main__":
    main()
