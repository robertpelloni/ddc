import os
import subprocess
import argparse
import sys

# Configuration
DIFFICULTIES = ["Beginner", "Easy", "Medium", "Expert", "Challenge"]
TYPES = ["dance-single", "dance-double"]


def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)


def prepare_data(packs_dir, work_dir):
    json_raw_dir = os.path.join(work_dir, "json_raw")
    json_filtered_dir = os.path.join(work_dir, "json_filtered")

    # 1. Extract raw JSON from packs
    print("--- Extracting JSON from packs ---")
    run_command(f"{sys.executable} dataset/extract_json.py {packs_dir} {json_raw_dir}")

    # 2. Filter into 10 buckets
    print("--- Filtering into buckets ---")

    if os.path.exists(json_raw_dir):
        pack_names = [
            d
            for d in os.listdir(json_raw_dir)
            if os.path.isdir(os.path.join(json_raw_dir, d))
        ]
    else:
        pack_names = []

    for pack_name in pack_names:
        pack_in_dir = os.path.join(json_raw_dir, pack_name)
        print(f"Filtering pack: {pack_name}")

        for chart_type in TYPES:
            for difficulty in DIFFICULTIES:
                bucket_name = f"{chart_type}_{difficulty}"
                bucket_out_dir = os.path.join(json_filtered_dir, bucket_name)

                run_command(
                    f"{sys.executable} dataset/filter_json.py {pack_in_dir} {bucket_out_dir} --chart_types {chart_type} --chart_difficulties {difficulty}"
                )

    # 3. Create dataset lists (train/test split)
    print("--- Creating dataset lists ---")
    for chart_type in TYPES:
        for difficulty in DIFFICULTIES:
            bucket_name = f"{chart_type}_{difficulty}"
            bucket_dir = os.path.join(json_filtered_dir, bucket_name)

            if not os.path.exists(bucket_dir):
                continue

            # Aggregate JSONs
            all_jsons = []
            for root, dirs, files in os.walk(bucket_dir):
                for f in files:
                    if f.endswith(".json"):
                        all_jsons.append(os.path.abspath(os.path.join(root, f)))

            if not all_jsons:
                print(f"Skipping empty bucket: {bucket_name}")
                continue

            import random

            random.shuffle(all_jsons)

            n = len(all_jsons)
            n_train = int(n * 0.8)
            n_valid = int(n * 0.1)

            # Ensure at least one training sample if data exists
            if n > 0 and n_train == 0:
                n_train = n
                n_valid = 0

            train_files = all_jsons[:n_train]
            valid_files = all_jsons[n_train : n_train + n_valid]
            test_files = all_jsons[n_train + n_valid :]

            with open(os.path.join(bucket_dir, "train.txt"), "w") as f:
                f.write("\n".join(train_files))
            with open(os.path.join(bucket_dir, "valid.txt"), "w") as f:
                f.write("\n".join(valid_files))
            with open(os.path.join(bucket_dir, "test.txt"), "w") as f:
                f.write("\n".join(test_files))

            print(
                f"Prepared {bucket_name}: {len(train_files)} train, {len(valid_files)} valid, {len(test_files)} test"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("packs_dir", type=str, help="Input packs directory")
    parser.add_argument(
        "work_dir", type=str, help="Working directory for processed data"
    )
    args = parser.parse_args()

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    prepare_data(args.packs_dir, args.work_dir)
