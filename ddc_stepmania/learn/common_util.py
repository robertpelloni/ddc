import numpy as np

def np_pad(x, pad_to, value=0, axis=-1):
    """
    Pads a numpy array along a specified axis to a target size.
    """
    assert x.shape[axis] <= pad_to
    pad = [(0, 0) for i in range(x.ndim)]
    pad[axis] = (0, pad_to - x.shape[axis])
    return np.pad(x, pad_width=pad, mode='constant', constant_values=value)

def make_onset_feature_context(song_features, frame_idx, radius):
    """
    Extracts a context window of features around a specific frame index.
    """
    nframes = song_features.shape[0]
    assert nframes > 0

    frame_idxs = range(frame_idx - radius, frame_idx + radius + 1)
    context = np.zeros((len(frame_idxs),) + song_features.shape[1:], dtype=song_features.dtype)
    for i, idx in enumerate(frame_idxs):
        if 0 <= idx < nframes:
            context[i] = song_features[idx]
        else:
            # Pad with zeros if the index is out of bounds
            context[i] = np.zeros_like(song_features[0])

    return context
