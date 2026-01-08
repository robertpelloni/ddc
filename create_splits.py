import os
import random
import sys


def create_dataset_splits(data_dir, output_dir, train_split=0.8, valid_split=0.1):
    """
    Reads .json files from a directory, shuffles them, and splits them into
    training, validation, and testing sets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Look for .json files now
    all_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".json")
    ]

    # Sort first to ensure deterministic shuffle if we seeded (we aren't seeding here but it's good practice before shuffle)
    all_files.sort()
    random.shuffle(all_files)

    total_files = len(all_files)
    print(f"Found {total_files} JSON files in {data_dir}")

    train_end = int(total_files * train_split)
    valid_end = int(total_files * (train_split + valid_split))

    train_files = all_files[:train_end]
    valid_files = all_files[train_end:valid_end]
    test_files = all_files[valid_end:]

    dataset_name = os.path.basename(os.path.normpath(data_dir))

    # Write absolute paths to the text files
    # The original script might have expected relative paths or something else,
    # but usually absolute paths are safest for downstream tools unless they are explicitly designed otherwise.
    # However, to be safe with how other scripts might read this, let's stick to what it was doing: os.path.join(data_dir, f) which produces the path provided.

    # We will use dataset_name prefix for the split files to avoid confusion if multiple datasets are used
    # e.g. ddr_official_train.txt

    train_out = os.path.join(output_dir, f"{dataset_name}_train.txt")
    valid_out = os.path.join(output_dir, f"{dataset_name}_valid.txt")
    test_out = os.path.join(output_dir, f"{dataset_name}_test.txt")

    with open(train_out, "w") as f:
        for item in train_files:
            f.write("%s\n" % os.path.abspath(item))

    with open(valid_out, "w") as f:
        for item in valid_files:
            f.write("%s\n" % os.path.abspath(item))

    with open(test_out, "w") as f:
        for item in test_files:
            f.write("%s\n" % os.path.abspath(item))

    print(f"Created splits in {output_dir}:")
    print(f"  Train: {len(train_files)} ({train_out})")
    print(f"  Valid: {len(valid_files)} ({valid_out})")
    print(f"  Test:  {len(test_files)}  ({test_out})")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_splits.py <data_dir> <output_dir>")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = sys.argv[2]

    create_dataset_splits(data_dir, output_dir)
