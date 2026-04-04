import os
import subprocess
import argparse
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packs_dir', type=str, help='Input packs directory')
    parser.add_argument('work_dir', type=str, help='Working directory for processed data and models')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()

    json_dir = os.path.join(args.work_dir, 'json')
    feats_dir = os.path.join(args.work_dir, 'feats')
    models_dir = os.path.join(args.work_dir, 'models')
    ffr_data_dir = os.path.join(args.work_dir, 'ffr_data')
    ffr_models_dir = os.path.join(args.work_dir, 'ffr_models')

    print("=== Step 1: Prepare Data ===")
    run_cmd(f"{sys.executable} scripts/prepare_data.py {args.packs_dir} {args.work_dir}")

    print("=== Step 2: Extract Features ===")
    all_jsons_list = os.path.join(args.work_dir, 'all_jsons.txt')
    
    # Use os.walk instead of 'find' for Windows compatibility and robust file listing
    with open(all_jsons_list, 'w') as f:
        json_filtered_path = os.path.join(args.work_dir, 'json_filtered')
        if os.path.exists(json_filtered_path):
            for root, dirs, files in os.walk(json_filtered_path):
                for file in files:
                    if file.endswith('.json'):
                        f.write(os.path.join(root, file) + '\n')
    
    if os.path.getsize(all_jsons_list) > 0:
        run_cmd(f"{sys.executable} learn/extract_feats_v2.py {all_jsons_list} {feats_dir} --jobs {args.jobs}")
    else:
        print("No JSON files found to extract features from.")

    print("=== Step 3: Train DDC Models ===")
    print("--- Training Onset Model ---")
    # Training onset on 'Hard' which is the standard Expert difficulty in StepMania
    onset_data_dir = os.path.join(args.work_dir, 'json_filtered', 'dance-single_Hard')
    if os.path.exists(onset_data_dir):
        # Using PyTorch training script instead of TF
        run_cmd(f"{sys.executable} scripts/train_pt.py --dataset_dir {onset_data_dir} --feats_dir {feats_dir} --out_dir {os.path.join(models_dir, 'onset')} --model_type onset --epochs 5")

    buckets = []
    # User requested 8 training runs (Easy, Medium, Hard, Challenge for Single/Double)
    # This excludes 'Beginner'.
    difficulties = ['Easy', 'Medium', 'Hard', 'Challenge']
    types = ['dance-single', 'dance-double']
    for t in types:
        for d in difficulties:
            buckets.append(f"{t}_{d}")

    for bucket in buckets:
        print(f"--- Training Sym Model: {bucket} ---")
        bucket_dir = os.path.join(args.work_dir, 'json_filtered', bucket)
        if not os.path.exists(os.path.join(bucket_dir, 'train.txt')):
            print(f"Skipping empty bucket {bucket}")
            continue

        out_dir = os.path.join(models_dir, bucket)
        # Using PyTorch training script
        run_cmd(f"{sys.executable} scripts/train_pt.py --dataset_dir {bucket_dir} --feats_dir {feats_dir} --out_dir {out_dir} --model_type sym --epochs 10")

    print("=== Step 4: Train FFR Model ===")
    if not os.path.exists(ffr_data_dir):
        os.makedirs(ffr_data_dir)

    ffr_csv = os.path.join(ffr_data_dir, 'dataset.csv')
    ffr_processed_charts = os.path.join(ffr_data_dir, 'processed_charts')

    ffr_script_dir = 'ffr-difficulty-model/scripts'
    env = os.environ.copy()
    env['PYTHONPATH'] = f"ffr-difficulty-model;{env.get('PYTHONPATH', '')}" 

    if os.path.exists(os.path.join(ffr_script_dir, 'make_dataset_from_sm.py')):
        print("Generating FFR Dataset CSV...")
        # Step 4a: Make dataset from SM (extract .chart files)
        run_cmd(f"{sys.executable} {os.path.join(ffr_script_dir, 'make_dataset_from_sm.py')} {args.packs_dir} {ffr_processed_charts}")
        
        # Step 4b: Build features from .chart files
        run_cmd(f"{sys.executable} {os.path.join(ffr_script_dir, 'build_features.py')} {ffr_processed_charts} {ffr_csv}")

        print("Training FFR Model...")
        # Step 4c: Train the model
        cmd = f"{sys.executable} {os.path.join(ffr_script_dir, 'train_model.py')} {ffr_csv} {ffr_models_dir}"
        subprocess.check_call(cmd, shell=True, env=env)

if __name__ == '__main__':
    main()
