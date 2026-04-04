import os
import sys
import numpy as np
import librosa
import mutagen
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3
import torch
import torch.nn.functional as F
import json
import requests
from PIL import Image
from io import BytesIO
import shutil

# Ensure local modules are importable
# Assuming this file is in <root>/infer/
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Add ffr-difficulty-model to path
FFR_DIR = os.path.join(REPO_ROOT, "ffr-difficulty-model")
if FFR_DIR not in sys.path:
    sys.path.append(FFR_DIR)

# Add ddc_onset to path
DDC_ONSET_DIR = os.path.join(REPO_ROOT, "ddc_onset")
if DDC_ONSET_DIR not in sys.path:
    sys.path.append(DDC_ONSET_DIR)

from learn.extract_feats_v2 import extract_mel_feats_librosa
from learn.models_pt import SymNet, OnsetNet
from learn.util import make_onset_feature_context

# Try importing simfile
try:
    import simfile
    from simfile.sm import SMChart, SMSimfile
except ImportError:
    print(
        "Warning: simfile library not found. Falling back to manual writing if possible, or failing."
    )
    simfile = None

# Try importing FFR predictor
try:
    from stepmania_difficulty_predictor.models.prediction_pipeline import \
        ModeAgnosticDifficultyPredictor as DifficultyPredictor
except ImportError:
    DifficultyPredictor = None

# Try importing ddc_onset
try:
    from ddc_onset.ddc_onset import util as ddc_util

    DDC_ONSET_AVAILABLE = True
except ImportError:
    DDC_ONSET_AVAILABLE = False
    print("Warning: ddc_onset not available. Install torch and resampy to use it.")

from learn.train_v2 import DEFAULT_CONFIG


class AutoChart:
    """
    Main class for automatic stepchart generation.
    """

    def __init__(self, models_dir, ffr_model_dir=None, google_key=None, cx=None):
        self.models_dir = models_dir
        self.ffr_predictor = None
        if DifficultyPredictor and ffr_model_dir:
            self.ffr_predictor = DifficultyPredictor(ffr_model_dir)

        self.google_key = google_key
        self.cx = cx

    def process_song(self, audio_fp, out_dir):
        print(f"Processing {audio_fp}...")

        artist, title, album = self.get_metadata(audio_fp)
        print(f"Metadata: {artist} - {title} ({album})")

        # Structure: out_dir/Album/Artist - Title/
        song_dir = os.path.join(out_dir, album, f"{artist} - {title}")
        if not os.path.exists(song_dir):
            os.makedirs(song_dir)

        dest_audio = os.path.join(song_dir, os.path.basename(audio_fp))
        if not os.path.exists(dest_audio):
            shutil.copy2(audio_fp, dest_audio)

        y, sr = librosa.load(audio_fp, sr=44100)

        print("Detecting BPM...")
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        bpms = [(0.0, float(tempo))]

        beats = None

        # Try ddc_onset first
        if DDC_ONSET_AVAILABLE:
            print("Running Onset Detection (ddc_onset)...")
            try:
                # ddc_onset expects 44100Hz mono audio
                # librosa load already gave us that (if mono=True default).
                # ddc_onset util handles resampling if needed.
                # It returns salience at 100Hz.

                # Check device
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                onset_salience = ddc_util.compute_onset_salience(y, sr, device=device)

                peaks = ddc_util.find_peaks(onset_salience)
                # Threshold - using 0.1 as a safe default, or maybe higher?
                # ddc_onset constants has DIFFICULTY_TO_THRESHOLD.
                # But we want *all* candidate onsets for the chart generation to filter?
                # Or just major beats?
                # DDC usually generates steps on onsets.
                # Let's use a low threshold to capture most musical events.

                thresholded_peaks = ddc_util.threshold_peaks(
                    onset_salience, peaks, threshold=0.1
                )

                # Convert frames (100Hz) to seconds
                beats = [p / 100.0 for p in thresholded_peaks]

            except Exception as e:
                print(f"ddc_onset failed: {e}. Falling back...")

        if beats is None:
            # Check for legacy Onset model
            onset_model_path = os.path.join(self.models_dir, "onset", "model.h5")
            if os.path.exists(onset_model_path):
                print("Running Onset Detection (Legacy Keras Model)...")
                # TODO: Implement inference for legacy model if needed,
                # but for now fallback to librosa if ddc_onset fails/missing
                # and legacy model code isn't fully ported in this snippet.
                # (The user asked to add ddc_onset, so that's the priority)
                pass

        if beats is None:
            print("Using Librosa beats.")
            beats = librosa.frames_to_time(beat_frames, sr=sr)

        charts = []

        difficulties = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]
        types = ["dance-single", "dance-double"]

        for chart_type in types:
            for difficulty in difficulties:
                model_name = f"{chart_type}_{difficulty}"
                model_path = os.path.join(self.models_dir, model_name, "model_10.pth")
                vocab_path = os.path.join(self.models_dir, model_name, "vocab.json")

                if not os.path.exists(model_path):
                    # Try other epochs if 10 doesn't exist
                    found = False
                    for epoch in range(9, 0, -1):
                        alt_path = os.path.join(self.models_dir, model_name, f"model_{epoch:02d}.pth")
                        if os.path.exists(alt_path):
                            model_path = alt_path
                            found = True
                            break
                    if not found:
                        continue

                print(f"Generating {difficulty} {chart_type}...")
                notes = self.generate_chart(audio_fp, model_path, vocab_path, beats, sr)

                charts.append(
                    {
                        "type": chart_type,
                        "difficulty": difficulty,
                        "meter": 1,
                        "notes": notes,
                    }
                )

        sm_fp = os.path.join(song_dir, f"{artist} - {title}.sm")
        self.write_sm(sm_fp, artist, title, os.path.basename(audio_fp), bpms, charts)

        if self.ffr_predictor:
            print("Rating difficulty...")
            try:
                preds = self.ffr_predictor.predict(sm_fp)
                for p in preds:
                    for c in charts:
                        if c["type"] == p.get("mode", "dance-single") and c[
                            "difficulty"
                        ] == p.get("difficulty"):
                            c["meter"] = float(p["predicted_difficulty"])
                            break

                self.write_sm(
                    sm_fp, artist, title, os.path.basename(audio_fp), bpms, charts
                )

            except Exception as e:
                print(f"FFR Rating failed: {e}")

        # Download images (fallback to embedded art)
        self.get_images(audio_fp, artist, title, song_dir)

    def get_metadata(self, audio_fp):
        try:
            audio = EasyID3(audio_fp)
            artist = audio.get("artist", ["Unknown Artist"])[0]
            title = audio.get("title", ["Unknown Title"])[0]
            album = audio.get("album", ["Unknown Album"])[0]
            return artist, title, album
        except:
            return "Unknown Artist", "Unknown Title", "Unknown Album"

    def generate_chart(self, audio_fp, model_path, vocab_path, beats, sr):
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        id_to_token = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab) + 1

        # Configuration for PyTorch model
        config = DEFAULT_CONFIG.copy()
        config["audio_context_radius"] = 1
        
        # Audio shape: (channels, context, freq)
        audio_shape = (3, config["audio_context_radius"] * 2 + 1, 80)
        n_other = 0

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SymNet(audio_shape, n_other, vocab_size, config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        song_feats = extract_mel_feats_librosa(audio_fp)
        frame_rate = 44100 / 512.0

        generated_tokens = []
        # Pad with 64 silence tokens
        current_seq = [vocab.get("<-1>", 0)] * 64

        notes_str = ""

        with torch.no_grad():
            state = None
            for i, beat_time in enumerate(beats):
                # For each beat (onset), predict the step
                input_syms = torch.tensor([current_seq[-64:]]).long().to(device)

                input_audio_list = []
                for j in range(64):
                    b_idx = i - 63 + j
                    if b_idx < 0:
                        f_idx = 0
                    else:
                        if b_idx < len(beats):
                            f_idx = int(beats[b_idx] * frame_rate)
                        else:
                            f_idx = song_feats.shape[0] - 1

                    feat = make_onset_feature_context(song_feats, f_idx, 1)
                    # Permute to (C, H, W) for PyTorch: (3, 3, 80)
                    feat = torch.from_numpy(feat).permute(2, 0, 1).float()
                    input_audio_list.append(feat)

                input_audio = torch.stack(input_audio_list).unsqueeze(0).to(device) # (1, 64, 3, 3, 80)
                input_other = torch.zeros((1, 64, 0)).to(device)

                # Forward pass
                # SymNet returns (logits, state)
                logits, state = model(input_syms, input_audio, input_other, state if i > 0 else None)
                
                # Take last prediction in sequence
                last_logits = logits[0, -1, :]
                token_id = torch.argmax(last_logits).item()
                token = id_to_token.get(token_id, "0000")

                generated_tokens.append(token)
                current_seq.append(token_id)

                # Format: note\n
                notes_str += token + "\n"
                # Add comma every 4 notes
                if (i + 1) % 4 == 0:
                    notes_str += ",\n"

        if not notes_str.endswith(";\n"):
            notes_str += ";"

        return notes_str

    def write_sm(self, sm_fp, artist, title, music_file, bpms, charts):
        if simfile:
            # Use simfile library
            sm = SMSimfile()
            sm["TITLE"] = title
            sm["ARTIST"] = artist
            sm["MUSIC"] = music_file
            sm["OFFSET"] = "0.0"
            sm["BPMS"] = f"{bpms[0][0]}={bpms[0][1]}"
            sm["STOPS"] = ""
            sm["BANNER"] = "banner.png"
            sm["BACKGROUND"] = "bg.png"

            for c in charts:
                chart = SMChart()
                chart.stepstype = c["type"]
                chart.difficulty = c["difficulty"]
                chart.meter = str(c["meter"])
                chart.radarvalues = "0.0,0.0,0.0,0.0,0.0"
                chart.notes = c["notes"]
                sm.charts.append(chart)

            with open(sm_fp, "w") as f:
                sm.serialize(f)
        else:
            # Fallback manual writing
            with open(sm_fp, "w") as f:
                f.write(f"#TITLE:{title};\n")
                f.write(f"#ARTIST:{artist};\n")
                f.write(f"#MUSIC:{music_file};\n")
                f.write(f"#OFFSET:0.0;\n")
                f.write(f"#BPMS:{bpms[0][0]}={bpms[0][1]};\n")
                f.write(f"#STOPS:;\n")
                f.write(f"#BANNER:banner.png;\n")
                f.write(f"#BACKGROUND:bg.png;\n")

                for c in charts:
                    f.write(f"//------------------\n")
                    f.write(f"#NOTES:\n")
                    f.write(f"     {c['type']}:\n")
                    f.write(f"     :\n")
                    f.write(f"     {c['difficulty']}:\n")
                    f.write(f"     {c['meter']}:\n")
                    f.write(f"     0.0,0.0,0.0,0.0,0.0:\n")

                    f.write(c["notes"])
                    f.write(f"\n;")

    def get_images(self, audio_fp, artist, title, out_dir):
        # 1. Try Google if keys provided
        if self.google_key and self.cx:
            print("Downloading images (Google API)...")
            self.download_images_google(artist, title, out_dir)
            return

        # 2. Try Embedded Art
        print("Checking for embedded art...")
        try:
            tags = ID3(audio_fp)
            for key in tags.keys():
                if key.startswith("APIC:"):
                    # Found art
                    data = tags[key].data

                    img = Image.open(BytesIO(data))
                    img.save(os.path.join(out_dir, "banner.png"))
                    img.save(os.path.join(out_dir, "bg.png"))
                    print("Extracted embedded art.")
                    return
        except Exception as e:
            # Not fatal
            pass

    def download_images_google(self, artist, title, out_dir):
        query = f"{artist} {title} album cover"
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={self.cx}&key={self.google_key}&searchType=image&num=1"
        try:
            res = requests.get(url).json()
            if "items" in res:
                img_url = res["items"][0]["link"]
                img_data = requests.get(img_url).content

                img = Image.open(BytesIO(img_data))
                img.save(os.path.join(out_dir, "banner.png"))
                img.save(os.path.join(out_dir, "bg.png"))
        except Exception as e:
            print(f"Image download failed: {e}")
