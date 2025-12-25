import librosa
import numpy as np
from pydub import AudioSegment

AudioSegment.converter = "/usr/bin/ffmpeg"

class FeatureExtractor:
    def __init__(self, fs=44100.0, nhop=512, nffts=[1024, 2048, 4096], mel_nband=80, mel_freqlo=27.5, mel_freqhi=16000.0):
        self.fs = fs
        self.nhop = nhop
        self.nffts = nffts
        self.mel_nband = mel_nband
        self.mel_freqlo = mel_freqlo
        self.mel_freqhi = mel_freqhi

    def extract_features(self, audio_fp, log_scale=True):
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_fp)
            # Resample to the target sampling rate
            audio = audio.set_frame_rate(self.fs)
            # Convert to mono
            audio = audio.set_channels(1)
            # Convert to numpy array and normalize
            y = np.array(audio.get_array_of_samples())
            if audio.sample_width == 2:
                y = y.astype(np.float32) / 32768.0
            elif audio.sample_width == 4:
                y = y.astype(np.float32) / 2147483648.0
            else:
                # Fallback for other sample widths
                y = y.astype(np.float32) / (2**(audio.sample_width * 8 - 1))

        except Exception as e:
            print(f"Error loading audio file {audio_fp} with pydub: {e}")
            return None

        feat_channels = []
        for n_fft in self.nffts:
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=self.fs,
                n_fft=n_fft,
                hop_length=self.nhop,
                n_mels=self.mel_nband,
                fmin=self.mel_freqlo,
                fmax=self.mel_freqhi
            )
            feat_channels.append(mel_spec)

        min_len = min(ch.shape[1] for ch in feat_channels)
        feat_channels = [ch[:, :min_len] for ch in feat_channels]
        
        feat_channels = np.stack(feat_channels, axis=-1)
        feat_channels = np.transpose(feat_channels, (1, 0, 2))

        if log_scale:
            feat_channels = librosa.power_to_db(feat_channels, ref=np.max)

        return feat_channels

if __name__ == '__main__':
    import argparse
    import pickle
    import json
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_fps', type=str, nargs='+', help='')
    parser.add_argument('--out_dir', type=str, required=True, help='')
    args = parser.parse_args()
    
    feature_extractor = FeatureExtractor()

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    for dataset_fp in args.dataset_fps:
        with open(dataset_fp, 'r') as f:
            json_fps = f.read().splitlines()

        for json_fp in json_fps:
            song_name = os.path.splitext(os.path.split(json_fp)[1])[0]
            print(f'Extracting feats from {song_name}')

            with open(json_fp, 'r') as json_f:
                meta = json.loads(json_f.read())
            
            music_fp = meta['music_fp']
            if not os.path.exists(music_fp):
                raise ValueError(f'No music for {json_fp}')

            song_feats = feature_extractor.extract_features(music_fp)

            feats_fp = os.path.join(args.out_dir, f'{song_name}.pkl')
            with open(feats_fp, 'wb') as f:
                pickle.dump(song_feats, f)
