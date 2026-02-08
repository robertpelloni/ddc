import os
import json
import numpy as np
import random
<<<<<<< HEAD

try:
    import tensorflow as tf

    Sequence = tf.keras.utils.Sequence
except ImportError:
    tf = None

    class Sequence:
        """Dummy class for when TensorFlow is not available."""

        pass


from learn.chart import Chart, OnsetChart, SymbolicChart


class DataGenerator(Sequence):
    def __init__(
        self, json_fps, feats_dir, batch_size, model_type, config, vocab_map=None
    ):
=======
import tensorflow as tf
from learn.chart import Chart, OnsetChart, SymbolicChart

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, json_fps, feats_dir, batch_size, model_type, config, vocab_map=None):
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
        self.json_fps = json_fps
        self.feats_dir = feats_dir
        self.batch_size = batch_size
        self.model_type = model_type
        self.config = config
        self.vocab_map = vocab_map

        self.charts = []
        self.valid_json_fps = []

        print(f"Loading metadata for {len(json_fps)} charts...")
        for json_fp in json_fps:
            try:
<<<<<<< HEAD
                with open(json_fp, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                json_name = os.path.splitext(os.path.basename(json_fp))[0]
                feat_fp = os.path.join(feats_dir, json_name + ".npy")
=======
                with open(json_fp, 'r') as f:
                    meta = json.load(f)

                json_name = os.path.splitext(os.path.basename(json_fp))[0]
                feat_fp = os.path.join(feats_dir, json_name + '.npy')
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

                if not os.path.exists(feat_fp):
                    continue

                valid_charts_in_song = []
<<<<<<< HEAD
                for chart_meta in meta["charts"]:
=======
                for chart_meta in meta['charts']:
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
                    valid_charts_in_song.append(chart_meta)

                if not valid_charts_in_song:
                    continue

                self.valid_json_fps.append(json_fp)

                for chart_meta in valid_charts_in_song:
<<<<<<< HEAD
                    self.charts.append(
                        {
                            "json_fp": json_fp,
                            "feat_fp": feat_fp,
                            "meta": meta,
                            "chart_meta": chart_meta,
                        }
                    )
=======
                    self.charts.append({
                        'json_fp': json_fp,
                        'feat_fp': feat_fp,
                        'meta': meta,
                        'chart_meta': chart_meta
                    })
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

            except Exception as e:
                print(f"Error loading {json_fp}: {e}")

<<<<<<< HEAD
        # If no valid charts loaded, this is a critical failure.
        if len(self.charts) == 0:
            print("WARNING: No valid charts loaded in DataGenerator!")
            # We must not proceed with empty charts or the generator will loop forever or crash.
            # But the __init__ just returns. The __getitem__ loop checks 'if not self.charts: break'

=======
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
        print(f"Loaded {len(self.charts)} valid charts.")
        self.indices = np.arange(len(self.charts))

    def __len__(self):
        return 1000

    def __getitem__(self, idx):
        batch_inputs = []
        batch_targets = []

<<<<<<< HEAD
        if self.model_type == "onset":
=======
        if self.model_type == 'onset':
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
            b_audio, b_other, b_y = [], [], []
        else:
            b_sym, b_audio, b_other, b_y = [], [], [], []

        for _ in range(self.batch_size):
            if not self.charts:
<<<<<<< HEAD
                break

            chart_info = np.random.choice(self.charts)

            try:
                if os.path.exists(chart_info["feat_fp"]):
                    song_feats = np.load(chart_info["feat_fp"], mmap_mode="r")
                else:
                    # Critical Failure: Missing Features
                    raise FileNotFoundError(
                        f"Feature file not found: {chart_info['feat_fp']}"
                    )

            except Exception as e:
                print(f"Error loading features for {chart_info['json_fp']}: {e}")
                # Skip this chart if loading fails
                # To do this effectively in the loop, we should probably pick another chart?
                # But for now, let's just return zeros to avoid crash, but log heavily.
                n_frames = 10000
                song_feats = np.zeros((n_frames, 80, 3), dtype=np.float32)

            song_meta = chart_info["meta"]
            chart_meta = chart_info["chart_meta"]

            fs = self.config.get("fs", 44100)
            nhop = self.config.get("nhop", 512)
            frame_rate = fs / float(nhop)

            if self.model_type == "onset":
                chart = OnsetChart(
                    song_meta,
                    song_feats,
                    frame_rate,
                    (
                        chart_meta["difficulty_coarse"],
                        chart_meta["difficulty_fine"],
                        chart_meta["type"],
                        chart_meta.get("desc_or_author"),
                    ),
                    chart_meta["notes"],
                )
=======
                 break

            chart_info = np.random.choice(self.charts)

            song_feats = np.load(chart_info['feat_fp'], mmap_mode='r')

            song_meta = chart_info['meta']
            chart_meta = chart_info['chart_meta']

            fs = self.config.get('fs', 44100)
            nhop = self.config.get('nhop', 512)
            frame_rate = fs / float(nhop)

            if self.model_type == 'onset':
                chart = OnsetChart(song_meta, song_feats, frame_rate,
                                   (chart_meta['difficulty_coarse'], chart_meta['difficulty_fine'], chart_meta['type'], chart_meta.get('desc_or_author')),
                                   chart_meta['notes'])
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

                if np.random.random() > 0.5 and len(chart.onsets) > 0:
                    frame = np.random.choice(list(chart.onsets))
                else:
                    frame = np.random.randint(0, chart.nframes)

<<<<<<< HEAD
                feats_audio, feats_other, y = chart.get_example(
                    frame,
                    dtype=np.float32,
                    time_context_radius=self.config.get("audio_context_radius", 7),
                )
=======
                feats_audio, feats_other, y = chart.get_example(frame, dtype=np.float32, time_context_radius=self.config.get('audio_context_radius', 7))
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
                b_audio.append(feats_audio)
                b_other.append(feats_other)
                b_y.append(y)

<<<<<<< HEAD
            elif self.model_type == "sym":
                chart = SymbolicChart(
                    song_meta,
                    song_feats,
                    frame_rate,
                    (
                        chart_meta["difficulty_coarse"],
                        chart_meta["difficulty_fine"],
                        chart_meta["type"],
                        chart_meta.get("desc_or_author"),
                    ),
                    chart_meta["notes"],
                )

                seq_len = self.config.get("rnn_nunroll", 32)
                syms, feats_other, feats_audio = chart.get_random_subsequence(
                    seq_len,
                    audio_time_context_radius=self.config.get(
                        "audio_context_radius", 1
                    ),
                )
=======
            elif self.model_type == 'sym':
                chart = SymbolicChart(song_meta, song_feats, frame_rate,
                                      (chart_meta['difficulty_coarse'], chart_meta['difficulty_fine'], chart_meta['type'], chart_meta.get('desc_or_author')),
                                      chart_meta['notes'])

                seq_len = self.config.get('rnn_nunroll', 32)
                syms, feats_other, feats_audio = chart.get_random_subsequence(seq_len, audio_time_context_radius=self.config.get('audio_context_radius', 1))
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

                sym_indices = []
                for s in syms:
                    if s not in self.vocab_map:
<<<<<<< HEAD
                        s = "0" * len(s)
=======
                         s = '0'*len(s)
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522
                    sym_indices.append(self.vocab_map.get(s, 0))

                input_syms = np.array(sym_indices[:-1])
                target_syms = np.array(sym_indices[1:])

                b_sym.append(input_syms)
                b_audio.append(feats_audio)
                b_other.append(feats_other)
                b_y.append(target_syms)

<<<<<<< HEAD
        if self.model_type == "onset":
            return [np.array(b_audio), np.array(b_other)], np.array(b_y)
        else:
            return [np.array(b_sym), np.array(b_audio), np.array(b_other)], np.array(
                b_y
            )
=======
        if self.model_type == 'onset':
            return [np.array(b_audio), np.array(b_other)], np.array(b_y)
        else:
            return [np.array(b_sym), np.array(b_audio), np.array(b_other)], np.array(b_y)
>>>>>>> origin/ddc-modernization-and-integration-14116118131799338522

    def on_epoch_end(self):
        pass
