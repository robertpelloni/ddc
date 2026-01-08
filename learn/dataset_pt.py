import torch
from torch.utils.data import Dataset
import numpy as np
from learn.data_gen import DataGenerator


class DDCDataset(Dataset):
    def __init__(
        self, json_fps, feats_dir, model_type, config, vocab_map=None, epoch_len=10000
    ):
        # reuse logic from DataGenerator to load charts
        self.gen = DataGenerator(json_fps, feats_dir, 1, model_type, config, vocab_map)
        self.model_type = model_type
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        # The DataGenerator generates a batch of size 1. We just take the first element.
        # This is a bit inefficient (wrapping in list then unwrapping), but saves rewriting the complex chart sampling logic for now.
        X, y = self.gen[0]

        # X is [audio, other] or [sym, audio, other]
        # y is target

        if self.model_type == "onset":
            # X: [ (1, context, 80, 3), (1, n_other) ]
            # y: (1,)
            if len(X[0]) == 0:
                # Safety for empty batches
                audio = torch.zeros(3, 7 * 2 + 1, 80, dtype=torch.float)
                other = torch.zeros(0, dtype=torch.float)
                target = torch.zeros(1, dtype=torch.float)
                return (audio, other), target

            audio = (
                torch.from_numpy(X[0][0]).permute(2, 0, 1).float()
            )  # (C, H, W) -> (3, context, 80)
            other = torch.from_numpy(X[1][0]).float()
            target = torch.tensor(y[0], dtype=torch.float32)
            return (audio, other), target

        elif self.model_type == "sym":
            # X: [ (1, seq), (1, seq, context, 80, 3), (1, seq, n_other) ]
            # y: (1, seq)
            if len(X[0]) == 0:  # Handle empty batch from generator
                # Return a dummy batch or skip.
                # Since we cannot easily skip in __getitem__, we'll raise an error or return zero-tensors that will be filtered.
                # Actually, let's fix the generator to ensure it yields data.
                # But for safety here:
                print(f"Warning: Empty batch from generator at idx {idx}")
                # Return zeros with correct shapes
                seq_len = 32  # config.get("rnn_nunroll", 32) hardcoded for fallback
                sym = torch.zeros(seq_len, dtype=torch.long)
                audio = torch.zeros(seq_len, 3, 1 * 2 + 1, 80, dtype=torch.float)
                other = torch.zeros(
                    seq_len, 0, dtype=torch.float
                )  # n_other unknown here
                target = torch.zeros(seq_len, dtype=torch.long)
                return (sym, audio, other), target

            sym = torch.from_numpy(X[0][0]).long()

            # Audio: (seq, context, 80, 3) -> (seq, 3, context, 80)

            audio = torch.from_numpy(X[1][0]).permute(0, 3, 1, 2).float()

            other = torch.from_numpy(X[2][0]).float()
            target = torch.from_numpy(y[0]).long()

            return (sym, audio, other), target


def collate_fn_onset(batch):
    # batch is list of ((audio, other), target)
    audio_list = [item[0][0] for item in batch]
    other_list = [item[0][1] for item in batch]
    target_list = [item[1] for item in batch]

    audio = torch.stack(audio_list)
    other = torch.stack(other_list)
    target = torch.stack(target_list)  # Removed unsqueeze(1) which was adding extra dim

    return (audio, other), target


def collate_fn_sym(batch):
    # batch is list of ((sym, audio, other), target)
    sym_list = [item[0][0] for item in batch]
    audio_list = [item[0][1] for item in batch]
    other_list = [item[0][2] for item in batch]
    target_list = [item[1] for item in batch]

    sym = torch.stack(sym_list)
    audio = torch.stack(audio_list)
    other = torch.stack(other_list)
    target = torch.stack(target_list)

    return (sym, audio, other), target
