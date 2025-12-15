import random
import os

def split_data(file_list_path, train_ratio=0.8, valid_ratio=0.1):
    """
    Splits a list of file paths into training, validation, and test sets.
    """
    if not os.path.exists(file_list_path):
        print(f"Error: File list not found at {file_list_path}")
        return

    with open(file_list_path, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]

    if not all_files:
        print("Error: The file list is empty.")
        return

    random.shuffle(all_files)

    num_files = len(all_files)
    num_train = int(num_files * train_ratio)
    num_valid = int(num_files * valid_ratio)
    
    train_files = all_files[:num_train]
    valid_files = all_files[num_train:num_train + num_valid]
    test_files = all_files[num_train + num_valid:]

    base_dir = os.path.dirname(file_list_path)

    with open(os.path.join(base_dir, 'train_files.txt'), 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(os.path.join(base_dir, 'valid_files.txt'), 'w') as f:
        f.write('\n'.join(valid_files))
        
    with open(os.path.join(base_dir, 'test_files.txt'), 'w') as f:
        f.write('\n'.join(test_files))

    print(f"Data split complete:")
    print(f"  Training set: {len(train_files)} files")
    print(f"  Validation set: {len(valid_files)} files")
    print(f"  Test set: {len(test_files)} files")

if __name__ == "__main__":
    split_data('all_sm_files.txt')
