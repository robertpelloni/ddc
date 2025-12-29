import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

def create_dummy_sym_model(vocab_size, output_path):
    # Simplified SymNetV2 architecture for testing

    # Sym input: (batch, seq_len)
    sym_in = layers.Input(shape=(None,), dtype='float32', name='sym_in')

    # Audio input: (batch, seq_len, context_frames, mel_bands, channels)
    # Autochart uses context radius 1 => 3 frames
    # Librosa mel feats => 80 bands, 3 channels (mels, delta, delta-delta)
    audio_in = layers.Input(shape=(None, 3, 80, 3), name='audio_in')

    # Other input: (batch, seq_len, 0)
    other_in = layers.Input(shape=(None, 0), name='other_in')

    # Simple embedding
    x_sym = layers.Embedding(vocab_size, 8)(sym_in)

    # Simple conv
    # TimeDistributed applies layer to each step in seq_len
    x_audio = layers.TimeDistributed(layers.Conv2D(4, (3,3), padding='same'))(audio_in)
    x_audio = layers.TimeDistributed(layers.Flatten())(x_audio)
    x_audio = layers.TimeDistributed(layers.Dense(8))(x_audio)

    x = layers.concatenate([x_sym, x_audio, other_in])
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(vocab_size, activation='softmax')(x)

    model = models.Model(inputs=[sym_in, audio_in, other_in], outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    # Save in .keras format to avoid warnings, though code loads h5
    model.save(output_path)
    return model

def create_dummy_vocab(mode, output_path):
    # mode: 'single' (4 chars) or 'double' (8 chars)
    # Create a small vocab
    vocab = {}
    vocab['<-1>'] = 0
    vocab['0' * (8 if mode == 'double' else 4)] = 1 # No-op

    # Add a few random notes
    for i in range(2, 10):
        if mode == 'double':
            note = f"{i%2}0000000"
        else:
            note = f"{i%2}000"
        vocab[note] = i

    with open(output_path, 'w') as f:
        json.dump(vocab, f)

    return len(vocab)
