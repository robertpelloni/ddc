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
<<<<<<< HEAD
    run_cmd(f"{sys.executable} scripts/prepare_data.py {args.packs_dir} {args.work_dir}")

    print("=== Step 2: Extract Features ===")
    all_jsons_list = os.path.join(args.work_dir, 'all_jsons.txt')
    # run_cmd(f"find {os.path.join(args.work_dir, 'json_filtered')} -name '*.json' > {all_jsons_list}")
    with open(all_jsons_list, 'w') as f:
        for root, dirs, files in os.walk(os.path.join(args.work_dir, 'json_filtered')):
            for file in files:
                if file.endswith('.json'):
                    f.write(os.path.join(root, file) + '\n')
    
    run_cmd(f"{sys.executable} learn/extract_feats_v2.py {all_jsons_list} {feats_dir} --jobs {args.jobs}")
=======
    run_cmd(f"python3 scripts/prepare_data.py {args.packs_dir} {args.work_dir}")

    print("=== Step 2: Extract Features ===")
    all_jsons_list = os.path.join(args.work_dir, 'all_jsons.txt')
    run_cmd(f"find {os.path.join(args.work_dir, 'json_filtered')} -name '*.json' > {all_jsons_list}")
    run_cmd(f"python3 learn/extract_feats_v2.py {all_jsons_list} {feats_dir} --jobs {args.jobs}")
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

    print("=== Step 3: Train DDC Models ===")
    print("--- Training Onset Model ---")
    onset_data_dir = os.path.join(args.work_dir, 'json_filtered', 'dance-single_Expert')
<<<<<<< HEAD
    run_cmd(f"{sys.executable} scripts/train_v2.py --dataset_dir {onset_data_dir} --feats_dir {feats_dir} --out_dir {os.path.join(models_dir, 'onset')} --model_type onset --epochs 5")
=======
    run_cmd(f"python3 scripts/train_v2.py --dataset_dir {onset_data_dir} --feats_dir {feats_dir} --out_dir {os.path.join(models_dir, 'onset')} --model_type onset --epochs 5")
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

    buckets = []
    difficulties = ['Beginner', 'Easy', 'Medium', 'Expert', 'Challenge']
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
<<<<<<< HEAD
        run_cmd(f"{sys.executable} scripts/train_v2.py --dataset_dir {bucket_dir} --feats_dir {feats_dir} --out_dir {out_dir} --model_type sym --epochs 10")
=======
        run_cmd(f"python3 scripts/train_v2.py --dataset_dir {bucket_dir} --feats_dir {feats_dir} --out_dir {out_dir} --model_type sym --epochs 10")
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

    print("=== Step 4: Train FFR Model ===")
    if not os.path.exists(ffr_data_dir):
        os.makedirs(ffr_data_dir)

    ffr_csv = os.path.join(ffr_data_dir, 'dataset.csv')

    ffr_script_dir = 'ffr-difficulty-model/scripts'
    env = os.environ.copy()
    env['PYTHONPATH'] = f"ffr-difficulty-model:{env.get('PYTHONPATH', '')}"

    if os.path.exists(os.path.join(ffr_script_dir, 'make_dataset_from_sm.py')):
        print("Generating FFR Dataset CSV...")
<<<<<<< HEAD
        cmd = f"{sys.executable} {os.path.join(ffr_script_dir, 'make_dataset_from_sm.py')} {args.packs_dir} {ffr_csv}"
        subprocess.check_call(cmd, shell=True, env=env)

        print("Training FFR Model...")
        cmd = f"{sys.executable} {os.path.join(ffr_script_dir, 'train_model.py')} {ffr_csv} {ffr_models_dir}"
=======
        cmd = f"python3 {os.path.join(ffr_script_dir, 'make_dataset_from_sm.py')} {args.packs_dir} {ffr_csv}"
        subprocess.check_call(cmd, shell=True, env=env)

        print("Training FFR Model...")
        cmd = f"python3 {os.path.join(ffr_script_dir, 'train_model.py')} {ffr_csv} {ffr_models_dir}"
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
        subprocess.check_call(cmd, shell=True, env=env)
    else:
        print("FFR scripts not found, skipping.")

    print("=== All Training Completed ===")

if __name__ == '__main__':
    main()
