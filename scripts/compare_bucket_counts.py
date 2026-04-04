import argparse
import os
from collections import OrderedDict
from datetime import date

SPLITS = ["train.txt", "valid.txt", "test.txt"]


def count_split_lines(bucket_dir: str):
    counts = OrderedDict()
    for split in SPLITS:
        fp = os.path.join(bucket_dir, split)
        if os.path.exists(fp):
            with open(fp, "r", encoding="utf-8") as f:
                counts[split] = sum(1 for line in f if line.strip())
        else:
            counts[split] = 0
    return counts


def collect_bucket_counts(json_filtered_dir: str):
    buckets = OrderedDict()
    if not os.path.exists(json_filtered_dir):
        return buckets
    for name in sorted(os.listdir(json_filtered_dir)):
        bucket_dir = os.path.join(json_filtered_dir, name)
        if os.path.isdir(bucket_dir):
            buckets[name] = count_split_lines(bucket_dir)
    return buckets


def render_markdown(before_dir: str, after_dir: str, before_counts, after_counts):
    today = date.today().isoformat()
    buckets = sorted(set(before_counts) | set(after_counts))
    lines = [
        "# Bucket Split Delta Audit",
        "",
        f"Date: {today}",
        "",
        "## Scope",
        "",
        f"- Before directory: `{before_dir}`",
        f"- After directory: `{after_dir}`",
        "",
        "## Per-Bucket Split Deltas",
        "",
        "| Bucket | Train before | Train after | Train delta | Valid before | Valid after | Valid delta | Test before | Test after | Test delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for bucket in buckets:
        before = before_counts.get(bucket, {k: 0 for k in SPLITS})
        after = after_counts.get(bucket, {k: 0 for k in SPLITS})
        bt, bv, bs = before["train.txt"], before["valid.txt"], before["test.txt"]
        at, av, a_s = after["train.txt"], after["valid.txt"], after["test.txt"]
        lines.append(
            f"| `{bucket}` | {bt} | {at} | {at-bt:+d} | {bv} | {av} | {av-bv:+d} | {bs} | {a_s} | {a_s-bs:+d} |"
        )

    lines += ["", "## Practical Highlights", ""]
    important = [
        "dance-single_Easy",
        "dance-single_Medium",
        "dance-single_Hard",
        "dance-single_Challenge",
        "dance-double_Easy",
        "dance-double_Medium",
        "dance-double_Hard",
        "dance-double_Challenge",
    ]
    for bucket in important:
        before = before_counts.get(bucket, {k: 0 for k in SPLITS})
        after = after_counts.get(bucket, {k: 0 for k in SPLITS})
        lines.append(
            f"- `{bucket}`: train {before['train.txt']} -> {after['train.txt']} ({after['train.txt']-before['train.txt']:+d}), "
            f"valid {before['valid.txt']} -> {after['valid.txt']} ({after['valid.txt']-before['valid.txt']:+d}), "
            f"test {before['test.txt']} -> {after['test.txt']} ({after['test.txt']-before['test.txt']:+d})"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- This report isolates the downstream training-input delta after `.ssc` support and refreshed bucket preparation.",
        "- It is useful for confirming that the expanded corpus meaningfully increases the exact split files consumed by the practical DDC training plan.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Compare bucket split counts between two prepared work directories.")
    parser.add_argument("--before", default="data/training_output/json_filtered")
    parser.add_argument("--after", default="data/ssc_refresh_work/json_filtered")
    parser.add_argument("--out", default="docs/BUCKET_SPLIT_DELTA_2026-04-04.md")
    args = parser.parse_args()

    before_counts = collect_bucket_counts(args.before)
    after_counts = collect_bucket_counts(args.after)
    markdown = render_markdown(args.before, args.after, before_counts, after_counts)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)


if __name__ == "__main__":
    main()
