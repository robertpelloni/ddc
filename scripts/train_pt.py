import sys
import os

# Add the parent directory (ddc root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from learn.models_pt import OnsetNet, SymNet
from learn.dataset_pt import DDCDataset, collate_fn_onset, collate_fn_sym
from learn.train_v2 import build_vocab, DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--feats_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--model_type", type=str, required=True, choices=["onset", "sym"]
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f"Using device: {args.device}")
    device = torch.device(args.device)

    with open(os.path.join(args.dataset_dir, "train.txt"), "r") as f:
        train_json_fps = [x.strip() for x in f.read().splitlines() if x.strip()]

    valid_json_fps = []
    if os.path.exists(os.path.join(args.dataset_dir, "valid.txt")):
        with open(os.path.join(args.dataset_dir, "valid.txt"), "r") as f:
            valid_json_fps = [x.strip() for x in f.read().splitlines() if x.strip()]

    config = DEFAULT_CONFIG.copy()

    if args.model_type == "onset":
        train_dataset = DDCDataset(train_json_fps, args.feats_dir, "onset", config)
        valid_dataset = (
            DDCDataset(valid_json_fps, args.feats_dir, "onset", config, epoch_len=100)
            if valid_json_fps
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_onset,
            num_workers=0,
        )  # workers=0 for simplicity/pickle safety
        valid_loader = (
            DataLoader(
                valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn_onset
            )
            if valid_dataset
            else None
        )

        # Determine input shape
        # (3, context*2+1, 80)
        audio_shape = (3, config["audio_context_radius"] * 2 + 1, 80)
        n_other = 0  # Updated inside DataGenerator logic if needed, but currently hardcoded 0 in train_v2 logic if no extra feats
        # Actually dataset_pt uses the same logic. Let's check n_other.
        # DataGenerator returns b_other. OnsetChart.get_example returns empty other if no diff flags.
        # BUT train_v2 sets n_other = 0.
        # Wait, OnsetChart.get_example populates feats_other based on kwargs like diff_feet_to_id.
        # DataGenerator default config doesn't set these. So n_other is 0.

        model = OnsetNet(audio_shape, n_other, config).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    elif args.model_type == "sym":
        config["audio_context_radius"] = 1
        vocab_map = build_vocab(train_json_fps)
        vocab_size = len(vocab_map) + 1
        print(f"Vocab Size: {vocab_size}")

        with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
            json.dump(vocab_map, f)

        train_dataset = DDCDataset(
            train_json_fps, args.feats_dir, "sym", config, vocab_map
        )
        valid_dataset = (
            DDCDataset(
                valid_json_fps, args.feats_dir, "sym", config, vocab_map, epoch_len=100
            )
            if valid_json_fps
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn_sym,
            num_workers=0,
        )
        valid_loader = (
            DataLoader(
                valid_dataset, batch_size=args.batch_size, collate_fn=collate_fn_sym
            )
            if valid_dataset
            else None
        )

        # (3, context*2+1, 80)
        audio_shape = (3, config.get("audio_context_radius", 1) * 2 + 1, 80)
        n_other = 0

        model = SymNet(audio_shape, n_other, vocab_size, config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for inputs, targets in pbar:
            # Move to device
            if args.model_type == "onset":
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device)
            else:
                inputs = [x.to(device) for x in inputs]
                targets = targets.to(device).view(
                    -1
                )  # Flatten for CrossEntropy? No, (B, Seq) -> (B*Seq)
                # For sym model, targets are (B, Seq). Logits are (B, Seq, Vocab).
                # CrossEntropyLoss expects (B, C, Seq) or (N, C).
                # Reshape logits to (B*Seq, Vocab)

            optimizer.zero_grad()
            outputs, _ = model(*inputs)

            if args.model_type == "sym":
                outputs = outputs.view(-1, outputs.size(-1))
                targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Acc
            if args.model_type == "onset":
                preds = (outputs > 0.5).float()
                acc = (preds == targets).float().mean()
            else:
                preds = torch.argmax(outputs, dim=1)
                acc = (preds == targets).float().mean()

            train_acc += acc.item()
            steps += 1
            pbar.set_postfix({"loss": train_loss / steps, "acc": train_acc / steps})

        # Validation
        if valid_loader:
            model.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_steps = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    if args.model_type == "onset":
                        inputs = [x.to(device) for x in inputs]
                        targets = targets.to(device)
                    else:
                        inputs = [x.to(device) for x in inputs]
                        targets = targets.to(device).view(-1)

                    outputs = model(*inputs)
                    
                    # Unpack if SymNet (returns tuple now)
                    if isinstance(outputs, tuple):
                         outputs = outputs[0]

                    if args.model_type == "sym":
                        outputs = outputs.view(-1, outputs.size(-1))

                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    if args.model_type == "onset":
                        preds = (outputs > 0.5).float()
                        acc = (preds == targets).float().mean()
                    else:
                        preds = torch.argmax(outputs, dim=1)
                        acc = (preds == targets).float().mean()

                    val_acc += acc.item()
                    val_steps += 1

            print(
                f"Val Loss: {val_loss / val_steps:.4f}, Val Acc: {val_acc / val_steps:.4f}"
            )

        # Save Checkpoint
        torch.save(
            model.state_dict(), os.path.join(args.out_dir, f"model_{epoch + 1:02d}.pth")
        )


if __name__ == "__main__":
    main()
