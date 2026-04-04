import argparse
import os
from datetime import date

TEXT_EXTS = {'.py', '.md', '.txt', '.bat', '.sh'}
IGNORE_DIRS = {'.git', '__pycache__', '.pytest_cache', '.ruff_cache', '.venv', 'node_modules'}


def iter_text_files(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in TEXT_EXTS:
                yield os.path.join(dirpath, filename)


def scan_file(path):
    has_conflict = False
    tf_hits = []
    h5_hits = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for idx, line in enumerate(f, start=1):
            if line.startswith('<<<<<<<') or line.startswith('=======') or line.startswith('>>>>>>>'):
                has_conflict = True
            lower = line.lower()
            if 'tensorflow' in lower or 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
                entry = f"{idx}: {line.strip()}"
                if 'tensorflow' in lower:
                    tf_hits.append(entry)
                if 'model.h5' in lower or 'train_v2.py' in lower or 'models_v2' in lower:
                    h5_hits.append(entry)
    return has_conflict, tf_hits, h5_hits


def render_markdown(root, conflicts, tf_refs, legacy_refs):
    today = date.today().isoformat()
    lines = [
        '# Repository Health Audit',
        '',
        f'Date: {today}',
        '',
        '## Scope',
        '',
        f'- Repository root: `{root}`',
        '',
        '## Summary',
        '',
        f'- Files containing unresolved merge-conflict markers: **{len(conflicts)}**',
        f'- Files containing TensorFlow references: **{len(tf_refs)}**',
        f'- Files containing legacy model-path / legacy-training references: **{len(legacy_refs)}**',
        '',
        '## Unresolved Merge-Conflict Markers',
        ''
    ]

    if conflicts:
        for path in conflicts:
            lines.append(f'- `{path}`')
    else:
        lines.append('- None found')

    lines += ['', '## TensorFlow Reference Hotspots', '']
    for path, hits in sorted(tf_refs.items()):
        lines.append(f'### `{path}`')
        lines.append('')
        for hit in hits[:20]:
            lines.append(f'- {hit}')
        if len(hits) > 20:
            lines.append(f'- ... {len(hits) - 20} more')
        lines.append('')

    lines += ['', '## Legacy Model / Training Reference Hotspots', '']
    for path, hits in sorted(legacy_refs.items()):
        lines.append(f'### `{path}`')
        lines.append('')
        for hit in hits[:20]:
            lines.append(f'- {hit}')
        if len(hits) > 20:
            lines.append(f'- ... {len(hits) - 20} more')
        lines.append('')

    lines += [
        '## Interpretation',
        '',
        '- The repository has advanced substantially, but it still contains unresolved merge-conflict markers in several files.',
        '- Legacy TensorFlow-oriented code paths and `.h5` references remain in parallel with the newer PyTorch-oriented work.',
        '- This does not block documentation work, corpus auditing, or targeted pipeline modernization, but it does mean a future cleanup pass is still warranted before calling the codebase fully normalized.',
        '',
        '## Recommended Next Cleanup Actions',
        '',
        '1. Resolve remaining merge-conflict markers in root metadata/docs and training-related modules.',
        '2. Decide which TensorFlow-era paths remain intentionally supported versus deprecated.',
        '3. Replace or clearly quarantine stale `.h5`/`models_v2`/`train_v2.py` references where the PyTorch path is now canonical for the active environment.',
        '4. After cleanup, rerun the repo-health audit to verify the blocker count drops toward zero.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Audit repository health for merge conflicts and legacy training references.')
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='docs/REPO_HEALTH_AUDIT_2026-04-04.md')
    args = parser.parse_args()

    conflicts = []
    tf_refs = {}
    legacy_refs = {}

    for path in iter_text_files(args.root):
        rel = os.path.relpath(path, args.root)
        has_conflict, tf_hits, legacy_hits = scan_file(path)
        if has_conflict:
            conflicts.append(rel)
        if tf_hits:
            tf_refs[rel] = tf_hits
        if legacy_hits:
            legacy_refs[rel] = legacy_hits

    markdown = render_markdown(args.root, sorted(conflicts), tf_refs, legacy_refs)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(markdown)
    print(markdown)


if __name__ == '__main__':
    main()
