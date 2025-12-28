#!/usr/bin/env python3
import argparse
import os
import sys

# Ensure local modules can be found
sys.path.append(os.getcwd())

try:
    from infer.autochart_lib import AutoChart
except ImportError:
    # If running directly from infer directory or installed weirdly
    sys.path.append(os.path.join(os.getcwd(), 'infer'))
    try:
        from autochart_lib import AutoChart
    except ImportError:
        print("Error: Could not import AutoChart class.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_paths', type=str, nargs='+', help='Input MP3/OGG files or directories')
    parser.add_argument('--out_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing trained models')
    parser.add_argument('--ffr_dir', type=str, help='Directory containing FFR models')
    parser.add_argument('--google_key', type=str, help='Google API Key')
    parser.add_argument('--cx', type=str, help='Google Custom Search Engine ID')
    parser.add_argument('--version', action='store_true', help='Show version information')

    args = parser.parse_args()

    if args.version:
        try:
            with open('VERSION.md', 'r') as f:
                print(f"DDC AutoChart v{f.read().strip()}")
        except FileNotFoundError:
            print("DDC AutoChart (Version unknown)")
        sys.exit(0)

    ac = AutoChart(args.models_dir, args.ffr_dir, args.google_key, args.cx)

    for input_path in args.input_paths:
        if os.path.isdir(input_path):
            # Batch mode
            print(f"Batch processing directory: {input_path}")
            files = []
            for root, dirs, filenames in os.walk(input_path):
                for filename in filenames:
                    if filename.lower().endswith(('.mp3', '.ogg', '.wav')):
                        files.append(os.path.join(root, filename))

            print(f"Found {len(files)} audio files in {input_path}.")
            for f in files:
                try:
                    ac.process_song(f, args.out_dir)
                except Exception as e:
                    print(f"Failed to process {f}: {e}")
        else:
            # Single file mode
            try:
                ac.process_song(input_path, args.out_dir)
            except Exception as e:
                print(f"Failed to process {input_path}: {e}")

if __name__ == '__main__':
    main()
