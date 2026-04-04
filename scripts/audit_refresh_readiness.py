import argparse
import glob
import os
from datetime import date


def count_nonempty_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def bucket_counts(json_filtered_dir):
    result = {}
    if not os.path.exists(json_filtered_dir):
        return result
    for bucket in sorted(os.listdir(json_filtered_dir)):
        bucket_dir = os.path.join(json_filtered_dir, bucket)
        if not os.path.isdir(bucket_dir):
            continue
        result[bucket] = {
            "train": count_nonempty_lines(os.path.join(bucket_dir, "train.txt")),
            "valid": count_nonempty_lines(os.path.join(bucket_dir, "valid.txt")),
            "test": count_nonempty_lines(os.path.join(bucket_dir, "test.txt")),
        }
    return result


def render_markdown(work_dir, counts, feat_count, models_dir, ffr_models_dir):
    today = date.today().isoformat()
    lines = [
        "# SSC Refresh Readiness Audit",
        "",
        f"Date: {today}",
        "",
        "## Scope",
        "",
        f"- Work directory: `{work_dir}`",
        "",
        "## Current Refresh State",
        "",
        f"- Filtered bucket directories present: **{len(counts)}**",
        f"- Extracted feature files present: **{feat_count}**",
        f"- Existing PyTorch model checkpoint directories under work dir: **{len([d for d in glob.glob(os.path.join(models_dir, '*')) if os.path.isdir(d)]) if os.path.exists(models_dir) else 0}**",
        f"- Existing FFR model files under work dir: **{len(glob.glob(os.path.join(ffr_models_dir, '*.p'))) if os.path.exists(ffr_models_dir) else 0}**",
        "",
        "## Bucket Counts",
        "",
        "| Bucket | Train | Valid | Test |",
        "|---|---:|---:|---:|",
    ]

    for bucket, split_counts in counts.items():
        lines.append(f"| `{bucket}` | {split_counts['train']} | {split_counts['valid']} | {split_counts['test']} |")

    lines += [
        "",
        "## Recommended Resume-Friendly Command",
        "",
        "```bash",
        f"python scripts/train_all.py data/raw/ddr_official {work_dir} --jobs 4 --skip_prepare --skip_feature_extraction --skip_existing_models",
        "```",
        "",
        "## Interpretation",
        "",
        "- This work directory is prepared specifically for the `.ssc`-inclusive refresh path.",
        "- The recommended command above is resume-friendly and avoids repeating already completed preparation/feature work.",
        "- If model artifacts are already present in the work directory, `--skip_existing_models` prevents redundant retraining for those buckets.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Audit refresh work-dir readiness for resume-friendly retraining.")
    parser.add_argument("--work-dir", default="data/ssc_refresh_work")
    parser.add_argument("--out", default="docs/SSC_REFRESH_READINESS_2026-04-04.md")
    args = parser.parse_args()

    json_filtered_dir = os.path.join(args.work_dir, "json_filtered")
    feats_dir = os.path.join(args.work_dir, "feats")
    models_dir = os.path.join(args.work_dir, "models")
    ffr_models_dir = os.path.join(args.work_dir, "ffr_models")

    counts = bucket_counts(json_filtered_dir)
    feat_count = len(glob.glob(os.path.join(feats_dir, "*.npy"))) if os.path.exists(feats_dir) else 0
    markdown = render_markdown(args.work_dir, counts, feat_count, models_dir, ffr_models_dir)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)


if __name__ == "__main__":
    main()
