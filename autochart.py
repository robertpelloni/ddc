import os
import argparse
import numpy as np
import librosa
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
import tensorflow as tf
import json
import subprocess
import requests
from PIL import Image
from io import BytesIO

from learn.extract_feats_v2 import extract_mel_feats_librosa
from learn.models_v2 import create_onset_model, create_sym_model
from learn.chart import Chart
from learn.util import make_onset_feature_context

import sys
sys.path.append(os.path.join(os.getcwd(), 'ffr-difficulty-model'))
try:
    from stepmania_difficulty_predictor.models.prediction_pipeline import DifficultyPredictor
except ImportError:
    print("Warning: Could not import DifficultyPredictor. FFR rating will be disabled.")
    DifficultyPredictor = None

class AutoChart:
    def __init__(self, models_dir, ffr_model_dir=None, google_key=None, cx=None):
        self.models_dir = models_dir
        self.ffr_predictor = None
        if DifficultyPredictor and ffr_model_dir:
             self.ffr_predictor = DifficultyPredictor(ffr_model_dir)

        self.google_key = google_key
        self.cx = cx

    def process_song(self, audio_fp, out_dir):
        print(f"Processing {audio_fp}...")

        artist, title = self.get_metadata(audio_fp)
        print(f"Metadata: {artist} - {title}")

        song_dir = os.path.join(out_dir, f"{artist} - {title}")
        if not os.path.exists(song_dir):
            os.makedirs(song_dir)

        dest_audio = os.path.join(song_dir, os.path.basename(audio_fp))
        if not os.path.exists(dest_audio):
            import shutil
            shutil.copy2(audio_fp, dest_audio)

        y, sr = librosa.load(audio_fp, sr=44100)
        duration = librosa.get_duration(y=y, sr=sr)

        print("Detecting BPM...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpms = [(0.0, float(tempo))]

        onset_model_path = os.path.join(self.models_dir, 'onset', 'model.h5')
        if os.path.exists(onset_model_path):
            print("Running Onset Detection...")
            pass
        else:
            print("Onset model not found, using Librosa beats.")
            beats = librosa.frames_to_time(beat_frames, sr=sr)

        charts = []

        difficulties = ['Beginner', 'Easy', 'Medium', 'Expert', 'Challenge']
        types = ['dance-single', 'dance-double']

        for chart_type in types:
            for difficulty in difficulties:
                model_name = f"{chart_type}_{difficulty}"
                model_path = os.path.join(self.models_dir, model_name, 'model.h5')
                vocab_path = os.path.join(self.models_dir, model_name, 'vocab.json')

                if not os.path.exists(model_path):
                    continue

                print(f"Generating {difficulty} {chart_type}...")
                notes = self.generate_chart(audio_fp, model_path, vocab_path, beats, sr)

                charts.append({
                    'type': chart_type,
                    'difficulty': difficulty,
                    'meter': 1,
                    'notes': notes
                })

        sm_fp = os.path.join(song_dir, f"{artist} - {title}.sm")
        self.write_sm(sm_fp, artist, title, os.path.basename(audio_fp), bpms, charts)

        if self.ffr_predictor:
            print("Rating difficulty...")
            try:
                preds = self.ffr_predictor.predict(sm_fp)
                for p in preds:
                    for c in charts:
                        if c['type'] == p.get('mode', 'dance-single') and c['difficulty'] == p.get('difficulty'):
                            c['meter'] = int(round(p['predicted_difficulty']))
                            break

                self.write_sm(sm_fp, artist, title, os.path.basename(audio_fp), bpms, charts)

            except Exception as e:
                print(f"FFR Rating failed: {e}")

        if self.google_key and self.cx:
            print("Downloading images...")
            self.download_images(artist, title, song_dir)

    def get_metadata(self, audio_fp):
        try:
            audio = EasyID3(audio_fp)
            artist = audio.get('artist', ['Unknown Artist'])[0]
            title = audio.get('title', ['Unknown Title'])[0]
            return artist, title
        except:
            return "Unknown Artist", "Unknown Title"

    def generate_chart(self, audio_fp, model_path, vocab_path, beats, sr):
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
        id_to_token = {v: k for k, v in vocab.items()}

        try:
            from learn.models_v2 import SymNetV2
            model = tf.keras.models.load_model(model_path, custom_objects={'SymNetV2': SymNetV2})
        except:
            model = tf.keras.models.load_model(model_path)

        # Extract features
        # We need song_feats aligned to beats.
        # Logic similar to chart.py but for inference.
        # make_onset_feature_context(song_feats, frame_idx, context_radius)

        song_feats = extract_mel_feats_librosa(audio_fp)

        # We need to map beats (seconds) to frame indices.
        # Librosa default hop 512, fs 44100.
        # frame = time * 44100 / 512
        frame_rate = 44100 / 512.0

        generated_tokens = []

        # Start seed (padding or start token?)
        # SymNet expects sequence.
        # We start with empty sequence (or padding).
        # We iterate beats.

        # For efficiency, we can batch if possible, but here greedy sequential.
        # State: sequence of previous tokens (indices).

        current_seq = [vocab.get('<-1>', 0)] * 64 # Assuming nunroll=64

        notes_str = ""

        for i, beat_time in enumerate(beats):
            frame_idx = int(beat_time * frame_rate)

            # Context
            # We need sequences of length 64.
            # Audio features: we need to extract for the PREVIOUS 64 steps?
            # Or is it window centered?
            # SymNet inputs:
            # 1. sym_seq (prev 64 symbols)
            # 2. audio_seq (prev 64 audio contexts)
            # 3. other_seq (prev 64 metadatas)

            # This requires we have beat times for previous 64 steps.
            # For inference, we know the beat times (from Onset/Beat detection).

            input_syms = np.array(current_seq[-64:])

            input_audio = []
            for j in range(64):
                # Beat index i - (64-1) + j
                b_idx = i - 63 + j
                if b_idx < 0:
                    # Pad audio? or use frame 0?
                    # Or beat times before 0?
                    # Let's use zeros or first frame.
                    f_idx = 0
                else:
                    if b_idx < len(beats):
                        f_idx = int(beats[b_idx] * frame_rate)
                    else:
                        f_idx = song_feats.shape[0] - 1

                feat = make_onset_feature_context(song_feats, f_idx, 1) # radius 1
                input_audio.append(feat)

            input_audio = np.array(input_audio)

            # Other feats (metadata) - assume 0 for now
            input_other = np.zeros((64, 0)) # Assuming 0 extra features

            # Predict
            # Model expects batch dim

            # input_syms shape (64,) -> (1, 64)
            # input_audio shape (64, 3, 80, 3) -> (1, 64, 3, 80, 3)
            # input_other -> (1, 64, 0)

            # Check model input shapes?
            # model.input_shape might help if we were debugging.

            preds = model.predict([
                input_syms[np.newaxis, :],
                input_audio[np.newaxis, :],
                input_other[np.newaxis, :]
            ], verbose=0)

            # preds shape (1, 64, vocab_size) (since return_sequences=True)
            # We want the prediction for the LAST step.
            last_pred = preds[0, -1, :]

            token_id = np.argmax(last_pred)
            token = id_to_token.get(token_id, "0000")

            generated_tokens.append(token)
            current_seq.append(token_id)

            # Formatting for SM
            # Usually SM is measure based. 4 beats per measure (usually).
            # We need to format "0000\n0000\n0000\n0000\n,\n"
            # Assuming beats are 4th notes.

            notes_str += token + "\n"
            if (i + 1) % 4 == 0:
                notes_str += ",\n"

        if not notes_str.endswith(";\n"):
             notes_str += ";"

        return notes_str

    def write_sm(self, sm_fp, artist, title, music_file, bpms, charts):
        with open(sm_fp, 'w') as f:
            f.write(f"#TITLE:{title};\n")
            f.write(f"#ARTIST:{artist};\n")
            f.write(f"#MUSIC:{music_file};\n")
            f.write(f"#OFFSET:0.0;\n")
            f.write(f"#BPMS:{bpms[0][0]}={bpms[0][1]};\n")
            f.write(f"#STOPS:;\n")

            for c in charts:
                f.write(f"//------------------\n")
                f.write(f"#NOTES:\n")
                f.write(f"     {c['type']}:\n")
                f.write(f"     :\n")
                f.write(f"     {c['difficulty']}:\n")
                f.write(f"     {c['meter']}:\n")
                f.write(f"     0.0,0.0,0.0,0.0,0.0:\n")

                f.write(c['notes'])
                f.write(f"\n;\n")

    def download_images(self, artist, title, out_dir):
        query = f"{artist} {title} album cover"
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={self.cx}&key={self.google_key}&searchType=image&num=1"
        try:
            res = requests.get(url).json()
            if 'items' in res:
                img_url = res['items'][0]['link']
                img_data = requests.get(img_url).content

                img = Image.open(BytesIO(img_data))
                img.save(os.path.join(out_dir, "banner.png"))
                img.save(os.path.join(out_dir, "bg.png"))
        except Exception as e:
            print(f"Image download failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('audio_fp', type=str, help='Input MP3/OGG file')
    parser.add_argument('--out_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing trained models')
    parser.add_argument('--ffr_dir', type=str, help='Directory containing FFR models')
    parser.add_argument('--google_key', type=str, help='Google API Key')
    parser.add_argument('--cx', type=str, help='Google Custom Search Engine ID')

    args = parser.parse_args()

    ac = AutoChart(args.models_dir, args.ffr_dir, args.google_key, args.cx)
    ac.process_song(args.audio_fp, args.out_dir)

if __name__ == '__main__':
    main()
