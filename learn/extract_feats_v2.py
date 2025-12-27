import os
import argparse
import json
import numpy as np
import librosa
from tqdm import tqdm
import multiprocessing
from functools import partial

def extract_mel_feats_librosa(music_fp, fs=44100, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, log_scale=True):
    try:
        y, sr = librosa.load(music_fp, sr=fs, mono=True)
    except Exception as e:
        print(f"Error loading {music_fp}: {e}")
        return None

    feat_channels = []
    for nfft in nffts:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=nfft, hop_length=nhop, n_mels=mel_nband, fmin=27.5, fmax=16000.0, power=1.0
        )
        feat_channels.append(S.T)

    min_len = min([f.shape[0] for f in feat_channels])
    feat_channels = [f[:min_len] for f in feat_channels]

    feats = np.stack(feat_channels, axis=-1)

    if log_scale:
        feats = np.log(feats + 1e-16)

    return feats.astype(np.float32)

def process_file(json_fp, out_dir, args):
    try:
        with open(json_fp, 'r') as f:
            meta = json.load(f)

        music_fp = meta['music_fp']
        if not os.path.exists(music_fp):
             print(f"Audio not found: {music_fp}")
             return

        json_name = os.path.splitext(os.path.basename(json_fp))[0]
        out_fp = os.path.join(out_dir, json_name + '.npy')

        if os.path.exists(out_fp):
            return

        feats = extract_mel_feats_librosa(
            music_fp,
            fs=args.fs,
            nhop=args.nhop,
            nffts=args.nffts,
            mel_nband=args.mel_nband
        )

        if feats is not None:
            np.save(out_fp, feats)

    except Exception as e:
        print(f"Failed to process {json_fp}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_list', type=str, help='Text file with list of JSON files')
    parser.add_argument('out_dir', type=str, help='Output directory for .npy features')
    parser.add_argument('--fs', type=int, default=44100)
    parser.add_argument('--nhop', type=int, default=512)
    parser.add_argument('--nffts', type=str, default='1024,2048,4096')
    parser.add_argument('--mel_nband', type=int, default=80)
    parser.add_argument('--jobs', type=int, default=4)

    args = parser.parse_args()
    args.nffts = [int(x) for x in args.nffts.split(',')]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(args.dataset_list, 'r') as f:
        json_fps = [x.strip() for x in f.read().splitlines() if x.strip()]

    print(f"Extracting features for {len(json_fps)} songs...")

    with multiprocessing.Pool(args.jobs) as pool:
        func = partial(process_file, out_dir=args.out_dir, args=args)
        list(tqdm(pool.imap_unordered(func, json_fps), total=len(json_fps)))

if __name__ == '__main__':
    main()
