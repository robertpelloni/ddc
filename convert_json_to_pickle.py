import json
import pickle
import os
import sys

def convert_json_to_pickle(file_list_path):
    """
    Reads a file list (e.g., train.txt), loads each JSON file listed,
    and saves it as a pickle file. Creates a new file list with .pkl paths.
    """
    if not os.path.exists(file_list_path):
        print(f"Error: File list not found at {file_list_path}")
        return

    base_dir = os.path.dirname(file_list_path)
    file_list_name = os.path.basename(file_list_path)
    new_file_list_name = file_list_name.replace('.txt', '_pkl.txt')
    new_file_list_path = os.path.join(base_dir, new_file_list_name)

    print(f"Converting files listed in {file_list_path}...")
    print(f"New file list will be saved to {new_file_list_path}")

    with open(file_list_path, 'r') as f_in, open(new_file_list_path, 'w') as f_out:
        for json_path_line in f_in:
            json_path = json_path_line.strip()
            if not json_path:
                continue

            pkl_path = os.path.splitext(json_path)[0] + '.pkl'

            try:
                with open(json_path, 'r') as f_json:
                    data = json.load(f_json)

                with open(pkl_path, 'wb') as f_pkl:
                    pickle.dump(data, f_pkl)

                f_out.write(pkl_path + '\n')
            except Exception as e:
                print(f"Could not process {json_path}: {e}")

    print("Conversion complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_json_to_pickle.py <path_to_file_list.txt>")
        sys.exit(1)
    
    file_list = sys.argv[1]
    convert_json_to_pickle(file_list)
