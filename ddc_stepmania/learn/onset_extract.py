import pickle as pickle
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from absl import app
from absl import flags

from onset_net import OnsetNet
from util import *

flags.DEFINE_string('data_txt_fp', '', 'Training dataset txt file with a list of pickled song files')
flags.DEFINE_string('feats_dir', '', 'Subdirectory containing the audio features')
flags.DEFINE_string('train_dir', '', 'Directory for temporary training files and model weights')
flags.DEFINE_string('train_ckpt_fp', '', 'Path to model weights')
flags.DEFINE_integer('context_radius', 7, 'Past and future context per training example')
flags.DEFINE_integer('feat_dim', 80, 'Number of features per frame')
flags.DEFINE_integer('nchannels', 3, 'Number of channels per frame')
flags.DEFINE_bool('z_normalize_coeffs', False, 'Whether or not to normalize coeffs to zero mean, unit variance per band per channel.')
flags.DEFINE_string('dense_layer_sizes', '256', 'Comma-separated list of dense layer sizes')
flags.DEFINE_integer('export_feature_layer', 0, 'Which dense layer to use for features')
flags.DEFINE_string('out_dir', '', 'Directory for output')

FLAGS = flags.FLAGS
dtype = tf.float32

BATCH_SIZE = 512

def test():
    print('Loading data...')
    with open(FLAGS.data_txt_fp, 'r') as f:
        pkl_fps = f.read().split()

    # Create model
    print('Creating model')
    dense_layer_sizes = [int(x) for x in FLAGS.dense_layer_sizes.split(',')]
    model = OnsetNet(mode='eval',
                     batch_size=BATCH_SIZE,
                     audio_context_radius=FLAGS.context_radius,
                     audio_nbands=FLAGS.feat_dim,
                     audio_nchannels=FLAGS.nchannels,
                     nfeats=0,
                     cnn_filter_shapes=[],
                     cnn_init=None,
                     cnn_pool=[],
                     cnn_rnn_zack=False,
                     rnn_cell_type=None,
                     rnn_size=0,
                     rnn_nlayers=0,
                     rnn_init=None,
                     rnn_nunroll=1,
                     rnn_keep_prob=1.0,
                     dnn_sizes=dense_layer_sizes,
                     dnn_init=None,
                     dnn_keep_prob=1.0,
                     dnn_nonlin='relu',
                     target_weight_strategy='seq',
                     grad_clip=None,
                     opt=None,
                     export_feat_name='dnn_{}'.format(FLAGS.export_feature_layer - 1))

    if FLAGS.z_normalize_coeffs:
        print('Normalizing data to zero mean unit var')
        with open(os.path.join(FLAGS.train_dir, 'test_mean_std.pkl'), 'rb') as f:
            mean_per_band, std_per_band = pickle.load(f)

    if not os.path.isdir(FLAGS.out_dir):
        os.makedirs(FLAGS.out_dir)

    with tf.Session() as sess:
        print('Restoring model weights from {}'.format(FLAGS.train_ckpt_fp))
        model.train_saver.restore(sess, FLAGS.train_ckpt_fp)

        for pkl_fp in tqdm(pkl_fps):
            with open(os.path.join(FLAGS.feats_dir, pkl_fp), 'rb') as f:
                song_features = pickle.load(f)

            nframes = song_features.shape[0]

            # Normalize data
            if FLAGS.z_normalize_coeffs:
                apply_z_norm([(None, song_features, None)], mean_per_band, std_per_band)

            song_context, _ = model.prepare_test(song_features, 0)
            song_export = []
            for i in range(0, nframes, BATCH_SIZE):
                batch_features = song_context[i:i + BATCH_SIZE]
                feed_dict = {
                    model.input_context: batch_features,
                    model.difficulty: np.zeros(batch_features.shape[0], dtype=np.float32),
                    model.dropout_keep_p: 1.0
                }
                batch_export = sess.run(model.export_features, feed_dict=feed_dict)
                song_export.append(batch_export)
            song_export = np.concatenate(song_export)

            out_pkl_name = os.path.split(pkl_fp)[1]
            out_pkl_fp = os.path.join(FLAGS.out_dir, out_pkl_name)
            with open(out_pkl_fp, 'wb') as f:
                pickle.dump(song_export, f)


def main(_):
    test()


if __name__ == '__main__':
    app.run(main)
