#!/usr/bin/env python3
"""Create a deployment-ready bundle from a completed SSC refresh work directory.

This script packages the local training outputs into a cleaner bundle layout for
AutoChart / ddc_server usage without requiring the entire work directory.

Expected source layout:
  <work_dir>/models/
  <work_dir>/ffr_models/

Output layout:
  <bundle_dir>/models/
    onset/
    dance-single_Easy/
    dance-single_Medium/
    dance-single_Hard/
    dance-single_Challenge/
    dance-double_Easy/
    dance-double_Medium/
    dance-double_Hard/
    dance-double_Challenge/
  <bundle_dir>/ffr_models/
    dance-single.p
    dance-double.p
  <bundle_dir>/MANIFEST.json
  <bundle_dir>/README.txt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PRACTICAL_BUCKETS = [
    "dance-single_Easy",
    "dance-single_Medium",
    "dance-single_Hard",
    "dance-single_Challenge",
    "dance-double_Easy",
    "dance-double_Medium",
    "dance-double_Hard",
    "dance-double_Challenge",
]


def find_latest_checkpoint(model_dir: Path) -> Path | None:
    checkpoints = sorted(model_dir.glob("model_*.pth"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def ensure_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def copy_file(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] copy {src} -> {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_manifest(work_dir: Path, latest_only: bool) -> Dict[str, object]:
    models_dir = work_dir / "models"
    ffr_dir = work_dir / "ffr_models"

    manifest: Dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_work_dir": str(work_dir),
        "latest_only": latest_only,
        "models": {},
        "ffr_models": {},
    }

    onset_dir = models_dir / "onset"
    onset_ckpts = sorted(p.name for p in onset_dir.glob("model_*.pth"))
    manifest["models"]["onset"] = {
        "checkpoint_count": len(onset_ckpts),
        "checkpoints": onset_ckpts if not latest_only else onset_ckpts[-1:],
    }

    for bucket in PRACTICAL_BUCKETS:
        bucket_dir = models_dir / bucket
        ckpts = sorted(p.name for p in bucket_dir.glob("model_*.pth"))
        has_vocab = (bucket_dir / "vocab.json").exists()
        manifest["models"][bucket] = {
            "checkpoint_count": len(ckpts),
            "checkpoints": ckpts if not latest_only else ckpts[-1:],
            "has_vocab": has_vocab,
        }

    for fp in sorted(ffr_dir.glob("*.p")):
        manifest["ffr_models"][fp.name] = {
            "size_bytes": fp.stat().st_size,
            "modified_utc": datetime.fromtimestamp(fp.stat().st_mtime, timezone.utc).isoformat(),
        }

    return manifest


def package_bundle(work_dir: Path, bundle_dir: Path, latest_only: bool, overwrite: bool, dry_run: bool) -> Dict[str, object]:
    models_dir = work_dir / "models"
    ffr_dir = work_dir / "ffr_models"

    ensure_exists(models_dir, "models directory")
    ensure_exists(ffr_dir, "FFR models directory")

    if bundle_dir.exists() and any(bundle_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Bundle directory is not empty: {bundle_dir}")
        if dry_run:
            print(f"[dry-run] remove existing bundle directory contents: {bundle_dir}")
        else:
            shutil.rmtree(bundle_dir)

    if not dry_run:
        bundle_dir.mkdir(parents=True, exist_ok=True)

    out_models_dir = bundle_dir / "models"
    out_ffr_dir = bundle_dir / "ffr_models"

    # Onset
    onset_dir = models_dir / "onset"
    ensure_exists(onset_dir, "onset directory")
    onset_ckpts = sorted(onset_dir.glob("model_*.pth"))
    if not onset_ckpts:
        raise FileNotFoundError(f"No onset checkpoints found in {onset_dir}")
    onset_to_copy = [find_latest_checkpoint(onset_dir)] if latest_only else onset_ckpts
    for src in onset_to_copy:
        copy_file(src, out_models_dir / "onset" / src.name, dry_run)

    # Sym buckets
    for bucket in PRACTICAL_BUCKETS:
        bucket_dir = models_dir / bucket
        ensure_exists(bucket_dir, f"bucket directory {bucket}")
        vocab = bucket_dir / "vocab.json"
        ensure_exists(vocab, f"vocab for {bucket}")
        latest = find_latest_checkpoint(bucket_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoints found for {bucket} in {bucket_dir}")

        copy_file(vocab, out_models_dir / bucket / vocab.name, dry_run)

        if latest_only:
            copy_file(latest, out_models_dir / bucket / latest.name, dry_run)
        else:
            for src in sorted(bucket_dir.glob("model_*.pth")):
                copy_file(src, out_models_dir / bucket / src.name, dry_run)

    # FFR
    ffr_files = sorted(ffr_dir.glob("*.p"))
    if not ffr_files:
        raise FileNotFoundError(f"No FFR .p files found in {ffr_dir}")
    for src in ffr_files:
        copy_file(src, out_ffr_dir / src.name, dry_run)

    manifest = build_manifest(work_dir, latest_only)

    readme = (
        "SSC refresh deployment bundle\n\n"
        "Usage examples:\n"
        f"  python autochart.py path/to/song.mp3 --models_dir {bundle_dir / 'models'} --ffr_dir {bundle_dir / 'ffr_models'}\n"
        f"  python infer/ddc_server.py --models_dir {bundle_dir / 'models'} --ffr_dir {bundle_dir / 'ffr_models'} --port 8080\n"
    )

    if dry_run:
        print(f"[dry-run] write manifest -> {bundle_dir / 'MANIFEST.json'}")
        print(f"[dry-run] write readme -> {bundle_dir / 'README.txt'}")
    else:
        (bundle_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (bundle_dir / "README.txt").write_text(readme, encoding="utf-8")

    return manifest


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package a completed SSC refresh work directory into a cleaner deployment bundle.")
    parser.add_argument("work_dir", help="Completed SSC refresh work directory (e.g. data/ssc_refresh_work)")
    parser.add_argument("bundle_dir", help="Output bundle directory")
    parser.add_argument("--latest_only", action="store_true", help="Copy only the latest checkpoint per model directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing non-empty bundle directory")
    parser.add_argument("--dry_run", action="store_true", help="Print planned actions without writing files")
    args = parser.parse_args(argv)

    work_dir = Path(args.work_dir).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()

    try:
        manifest = package_bundle(
            work_dir=work_dir,
            bundle_dir=bundle_dir,
            latest_only=args.latest_only,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Bundle packaging plan complete.")
    print(json.dumps({
        "bundle_dir": str(bundle_dir),
        "latest_only": args.latest_only,
        "model_dirs": list(manifest["models"].keys()),
        "ffr_models": list(manifest["ffr_models"].keys()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
