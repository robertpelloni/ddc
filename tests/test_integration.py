import unittest
import os
import shutil
import tempfile
import sys
import numpy as np
import wave
import struct

# Add root to path
sys.path.append(os.getcwd())

from infer.autochart_lib import AutoChart
from tests.test_utils import create_dummy_sym_model, create_dummy_vocab

class TestAutoChartIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, 'models')
        self.out_dir = os.path.join(self.test_dir, 'output')
        os.makedirs(self.models_dir)
        os.makedirs(self.out_dir)

        # Create Dummy Models
        # Single Expert
        self.setup_model('dance-single_Expert', 'single')
        # Double Expert
        self.setup_model('dance-double_Expert', 'double')

        # Create Dummy Audio (1 sec sine wave)
        self.audio_path = os.path.join(self.test_dir, 'test_audio.wav')
        self.create_dummy_wav(self.audio_path)

    def setup_model(self, model_name, mode):
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir)
        vocab_path = os.path.join(model_dir, 'vocab.json')
        model_path = os.path.join(model_dir, 'model.h5')

        vocab_size = create_dummy_vocab(mode, vocab_path)
        create_dummy_sym_model(vocab_size, model_path)

    def create_dummy_wav(self, path):
        rate = 44100
        duration = 5 # seconds
        frequency = 440.0

        frames = []
        for i in range(int(rate * duration)):
            value = int(32767.0 * np.sin(2.0 * np.pi * frequency * i / rate))
            frames.append(struct.pack('<h', value))

        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_end_to_end_generation(self):
        # Initialize AutoChart
        # We Mock ffr_model_dir as None to skip it (requires submodule init)
        # We Mock google_key as None
        ac = AutoChart(self.models_dir, ffr_model_dir=None)

        # Run process
        ac.process_song(self.audio_path, self.out_dir)

        # Verify Output Structure
        # out_dir/Unknown Album/Unknown Artist - Unknown Title/
        song_dir = os.path.join(self.out_dir, "Unknown Album", "Unknown Artist - Unknown Title")
        self.assertTrue(os.path.isdir(song_dir))

        sm_path = os.path.join(song_dir, "Unknown Artist - Unknown Title.sm")
        self.assertTrue(os.path.exists(sm_path))

        # Read SM file and verify contents
        with open(sm_path, 'r') as f:
            content = f.read()

        # Check for presence of charts
        self.assertIn("dance-single:", content)
        self.assertIn("dance-double:", content)

        # Check specific note lengths in the raw content
        # Parse manually since we might not have simfile installed in test env (though we do)
        lines = content.splitlines()

        in_double = False
        found_double_note = False

        for line in lines:
            line = line.strip()
            if "dance-double:" in line:
                in_double = True
            elif "dance-single:" in line:
                in_double = False

            # Identify a note line (sequence of 0123M)
            # Avoid tags #...:;
            if not line.startswith('#') and not line.startswith('/') and not line.endswith(':'):
                # If it looks like a note line
                # Remove comma/semicolon
                clean_line = line.replace(',', '').replace(';', '')
                if len(clean_line) > 0 and clean_line[0] in '01234M':
                    if in_double:
                        # Should be 8 chars
                        if len(clean_line) == 8:
                            found_double_note = True
                        else:
                            self.fail(f"Found non-8-char note in dance-double chart: {clean_line}")

        self.assertTrue(found_double_note, "Did not find any 8-char notes in dance-double chart")

if __name__ == '__main__':
    unittest.main()
