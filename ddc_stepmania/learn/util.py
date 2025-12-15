import pickle as pickle
from ddc_stepmania.learn.chart import OnsetChart
from ddc_stepmania.infer.extract_feats import FeatureExtractor
import os
import math
import sys

import unicodedata

import numpy as np
import tensorflow.compat.v1 as tf
from scipy.signal import argrelextrema

def variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False,
                                 seed=None, dtype=tf.float32):
    if not dtype.is_floating:
        raise TypeError('Cannot create initializer for non-floating point type.')
    if mode not in ['FAN_IN', 'FAN_OUT', 'FAN_AVG']:
        raise TypeError('Unknow mode %s [FAN_IN, FAN_OUT, FAN_AVG]', mode)

    def _initializer(shape, dtype=dtype, partition_info=None):
        f_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
        f_out = shape[-1]
        if mode == 'FAN_IN':
            n = f_in
        elif mode == 'FAN_OUT':
            n = f_out
        else:
            n = (f_in + f_out) / 2.0

        if uniform:
            limit = np.sqrt(3.0 * factor / n)
            return tf.random_uniform(shape, -limit, limit,
                                     dtype, seed=seed)
        else:
            trunc_stddev = np.sqrt(1.3 * factor / n)
            return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype,
                                       seed=seed)

    return _initializer

def load_id_dict(id_dict_fp):
    with open(id_dict_fp, 'r') as f:
        id_dict = {k:int(i) for k,i in [x.split(',') for x in f.read().splitlines()]}
        if '' in id_dict:
            id_dict[None] = id_dict['']
            del id_dict['']
        return id_dict

def ez_name(x):
    x = ''.join(x.strip().split())
    x_clean = []
    for char in x:
        if char.isalnum():
            x_clean.append(char)
        else:
            x_clean.append('_')
    return ''.join(x_clean)

def stride_csv_arg_list(arg, stride, cast=int):
    assert stride > 0
    l = [x.strip() for x in arg.split(',') if x.strip()]
    l = [cast(x) for x in l]
    assert len(l) % stride == 0
    result = []
    for i in range(0, len(l), stride):
        if stride == 1:
            subl = l[i]
        else:
            subl = tuple(l[i:i + stride])
        result.append(subl)
    return result

def open_dataset_fps(*args):
    datasets = []
    feature_extractor = FeatureExtractor()
    frame_rate = feature_extractor.fs / feature_extractor.nhop

    for data_fp in args:
        if not data_fp:
            datasets.append([])
            continue

        with open(data_fp, 'r') as f:
            song_fps = f.read().split()
        dataset = []
        for song_fp in song_fps:
            try:
                with open(song_fp, 'rb') as f:
                    song_data = pickle.load(f)

                music_path = song_data.get('music_fp')
                if not music_path or not os.path.exists(music_path):
                    print(f"Warning: Music file not found for {song_fp}, skipping.")
                    continue
                
                song_features = feature_extractor.extract_features(music_path)
                if song_features is None:
                    print(f"Warning: Could not extract features for {music_path}, skipping.")
                    continue

                charts = []
                for chart_dict in song_data['charts']:
                    metadata = (
                        chart_dict.get('difficulty_coarse', 'N/A'),
                        chart_dict.get('difficulty_fine', 0),
                        chart_dict.get('type', 'N/A'),
                        chart_dict.get('desc_or_author', '')
                    )
                    annotations = chart_dict.get('notes', [])
                    
                    if not annotations:
                        continue

                    chart_obj = OnsetChart(song_data, song_features, frame_rate, metadata, annotations)
                    charts.append(chart_obj)
                
                if not charts:
                    continue

                dataset.append((song_data, song_features, charts))
            except Exception as e:
                print(f"Error processing file {song_fp}: {e}")
                
        datasets.append(dataset)
    return datasets[0] if len(datasets) == 1 else datasets

def select_channels(dataset, channels):
    for i, (song_metadata, song_features, song_charts) in enumerate(dataset):
        song_features_selected = song_features[:, :, channels]
        dataset[i] = (song_metadata, song_features_selected, song_charts)
        for chart in song_charts:
            chart.song_features = song_features_selected

def apply_z_norm(dataset, mean_per_band, std_per_band):
    for i, (song_metadata, song_features, song_charts) in enumerate(dataset):
        song_features_z = song_features - mean_per_band
        song_features_z /= std_per_band
        dataset[i] = (song_metadata, song_features_z, song_charts)
        for chart in song_charts:
            chart.song_features = song_features_z

def calc_mean_std_per_band(dataset):
    mean_per_band_per_song = [np.mean(song_features, axis=0) for _, song_features, _ in dataset]
    std_per_band_per_song = [np.std(song_features, axis=0) for _, song_features, _ in dataset]
    mean_per_band = np.mean(np.array(mean_per_band_per_song), axis=0)
    std_per_band = np.mean(np.array(std_per_band_per_song), axis=0)

    return mean_per_band, std_per_band

def flatten_dataset_to_charts(dataset):
    return [item for sublist in [x[2] for x in dataset] for item in sublist]

def filter_chart_type(charts, chart_type):
    return [x for x in charts if x.get_type() == chart_type]

def find_pred_onsets(scores, window):
    if window.shape[0] > 0:
        onset_function = np.convolve(scores, window, mode='same')
    else:
        onset_function = scores
    # see page 592 of "Universal onset detection with bidirectional long short-term memory neural networks"
    maxima = argrelextrema(onset_function, np.greater_equal, order=1)[0]
    return set(list(maxima))

def align_onsets_to_sklearn(true_onsets, pred_onsets, scores, tolerance=0):
    # Build one-to-many dicts of candidate matches
    true_to_pred = {}
    pred_to_true = {}
    for true_idx in true_onsets:
        true_to_pred[true_idx] = []
        for pred_idx in range(true_idx - tolerance, true_idx + tolerance + 1):
            if pred_idx in pred_onsets:
                true_to_pred[true_idx].append(pred_idx)
                if pred_idx not in pred_to_true:
                    pred_to_true[pred_idx] = []
                pred_to_true[pred_idx].append(true_idx)

    # Create alignments
    true_to_pred_confidence = {}
    pred_idxs_used = set()
    for pred_idx, true_idxs in pred_to_true.items():
        true_idx_use = true_idxs[0]
        if len(true_idxs) > 1:
            for true_idx in true_idxs:
                if len(true_to_pred[true_idx]) == 1:
                    true_idx_use = true_idx
                    break
        true_to_pred_confidence[true_idx_use] = scores[pred_idx]
        assert pred_idx not in pred_idxs_used
        pred_idxs_used.add(pred_idx)

    # Create confidence list
    y_true = np.zeros_like(scores)
    y_true[list(true_onsets)] = 1.0
    y_scores = np.zeros_like(scores)
    for true_idx, confidence in true_to_pred_confidence.items():
        y_scores[true_idx] = confidence

    # Add remaining false positives
    for fp_idx in pred_onsets - pred_idxs_used:
        y_scores[fp_idx] = scores[fp_idx]

    # Truncate predictions to annotated range
    first_onset = min(true_onsets)
    last_onset = max(true_onsets)

    return y_true[first_onset:last_onset + 1], y_scores[first_onset:last_onset + 1]
