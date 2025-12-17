import os
import argparse
import json
import numpy as np
import tensorflow as tf
from learn.models_v2 import create_onset_model, create_sym_model
from learn.data_gen import DataGenerator

# Default Hyperparams
DEFAULT_CONFIG = {
    'fs': 44100,
    'nhop': 512,
    'audio_context_radius': 7,
    'cnn_filter_shapes': [(3, 3, 16), (3, 3, 32), (3, 3, 64)],
    'cnn_pool': [(1, 2), (1, 2), (1, 2)],
    'rnn_size': 256,
    'rnn_nlayers': 2,
    'dnn_sizes': [128, 64],
    'sym_embedding_size': 32,
    'rnn_nunroll': 64
}

def build_vocab(json_fps):
    vocab = set()
    print("Building vocabulary...")
    for json_fp in json_fps:
        try:
            with open(json_fp, 'r') as f:
                meta = json.load(f)
            for chart in meta['charts']:
                for _, _, _, note in chart['notes']:
                    vocab.add(note)
        except:
            pass

    vocab = sorted(list(vocab))
    token_to_id = {t: i+1 for i, t in enumerate(vocab)}
    token_to_id['<PAD>'] = 0

    for i in range(10):
        token_to_id[f'<-{i+1}>'] = len(token_to_id)

    return token_to_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing train.txt, valid.txt and json files')
    parser.add_argument('--feats_dir', type=str, required=True, help='Directory containing .npy audio features')
    parser.add_argument('--out_dir', type=str, required=True, help='Directory to save model checkpoints')
    parser.add_argument('--model_type', type=str, required=True, choices=['onset', 'sym'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(os.path.join(args.dataset_dir, 'train.txt'), 'r') as f:
        train_json_fps = [x.strip() for x in f.read().splitlines() if x.strip()]

    valid_json_fps = []
    if os.path.exists(os.path.join(args.dataset_dir, 'valid.txt')):
        with open(os.path.join(args.dataset_dir, 'valid.txt'), 'r') as f:
            valid_json_fps = [x.strip() for x in f.read().splitlines() if x.strip()]

    config = DEFAULT_CONFIG.copy()

    if args.model_type == 'onset':
        train_gen = DataGenerator(train_json_fps, args.feats_dir, args.batch_size, 'onset', config)
        valid_gen = DataGenerator(valid_json_fps, args.feats_dir, args.batch_size, 'onset', config) if valid_json_fps else None

        audio_shape = (config['audio_context_radius']*2 + 1, 80, 3)
        n_other = 0

        model = create_onset_model(audio_shape, 0, config)

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        model.fit(train_gen, validation_data=valid_gen, epochs=args.epochs,
                  callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, 'model_{epoch:02d}.h5'))])

    elif args.model_type == 'sym':
        # Override radius for SymNet
        config['audio_context_radius'] = 1

        vocab_map = build_vocab(train_json_fps)
        vocab_size = len(vocab_map) + 1
        print(f"Vocab Size: {vocab_size}")

        with open(os.path.join(args.out_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab_map, f)

        train_gen = DataGenerator(train_json_fps, args.feats_dir, args.batch_size, 'sym', config, vocab_map)
        valid_gen = DataGenerator(valid_json_fps, args.feats_dir, args.batch_size, 'sym', config, vocab_map) if valid_json_fps else None

        seq_len = config['rnn_nunroll']
        audio_shape = (config.get('audio_context_radius', 1)*2 + 1, 80, 3)

        n_other = 0

        model = create_sym_model(seq_len, audio_shape, n_other, vocab_size, config)

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_gen, validation_data=valid_gen, epochs=args.epochs,
                  callbacks=[tf.keras.callbacks.ModelCheckpoint(os.path.join(args.out_dir, 'model_{epoch:02d}.h5'))])

if __name__ == '__main__':
    main()
