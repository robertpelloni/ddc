import numpy as np

with open('data/ddr_official.txt', 'r') as f:
    lines = f.readlines()

np.random.shuffle(lines)

train_split = int(0.8 * len(lines))
valid_split = int(0.9 * len(lines))

train_lines = lines[:train_split]
valid_lines = lines[train_split:valid_split]
test_lines = lines[valid_split:]

with open('data/train.txt', 'w') as f:
    f.writelines(train_lines)

with open('data/valid.txt', 'w') as f:
    f.writelines(valid_lines)

with open('data/test.txt', 'w') as f:
    f.writelines(test_lines)
