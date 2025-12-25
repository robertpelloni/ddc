import os
import random

def create_dataset_splits(data_dir, output_dir, train_split=0.8, valid_split=0.1):
    """
    Reads .pkl files from a directory, shuffles them, and splits them into
    training, validation, and testing sets.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    random.shuffle(all_files)

    train_end = int(len(all_files) * train_split)
    valid_end = int(len(all_files) * (train_split + valid_split))

    train_files = all_files[:train_end]
    valid_files = all_files[train_end:valid_end]
    test_files = all_files[valid_end:]

    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for item in train_files:
            f.write("%s\n" % item)

    with open(os.path.join(output_dir, 'valid.txt'), 'w') as f:
        for item in valid_files:
            f.write("%s\n" % item)

    with open(os.path.join(output_dir, 'test.txt'), 'w') as f:
        for item in test_files:
            f.write("%s\n" % item)

if __name__ == '__main__':
    create_dataset_splits('train.json', '.')
