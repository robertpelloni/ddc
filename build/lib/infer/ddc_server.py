import os
import shutil
import tempfile
import uuid
import zipfile
from flask import Flask, request, send_from_directory, send_file
import argparse

# Use the new AutoChart library
try:
    from infer.autochart_lib import AutoChart
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from infer.autochart_lib import AutoChart

_FRONTEND_DIST_DIR = 'frontend'
app = Flask(
    __name__,
    static_url_path='',
    static_folder=_FRONTEND_DIST_DIR)

@app.route('/')
def index():
    return send_from_directory(_FRONTEND_DIST_DIR, 'index.html')

@app.route('/choreograph', methods=['POST'])
def choreograph():
    uploaded_file = request.files.get('audio_file')
    if uploaded_file is None or len(uploaded_file.filename) == 0:
        return 'Audio file required', 400

    filename = uploaded_file.filename
    ext = os.path.splitext(filename)[1]

    # Metadata from form or file
    song_artist = request.form.get('song_artist', 'Unknown Artist')
    song_title = request.form.get('song_title', 'Unknown Title')

    # DDC v2 generates all difficulties by default, so we ignore diff_coarse for generation
    # but we could filter the output if needed. For now, let's generate everything.

    out_dir = tempfile.mkdtemp()

    try:
        # Save uploaded audio
        song_id = uuid.uuid4()
        song_fp = os.path.join(out_dir, f"{song_id}{ext}")
        uploaded_file.save(song_fp)

        # Initialize AutoChart
        # We need to know where models are.
        # Using global args passed at startup
        ac = AutoChart(
            models_dir=ARGS.models_dir,
            ffr_model_dir=ARGS.ffr_dir,
            google_key=ARGS.google_key,
            cx=ARGS.cx
        )

        # Process
        # process_song expects an audio file path and an output directory.
        # It creates a subdirectory structure out_dir/Album/Artist-Title/
        # We want to capture that output.

        ac.process_song(song_fp, out_dir)

        # Find the generated content
        # It should be deeply nested.
        # Let's find the .sm file and the audio

        content_dir = None
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith('.sm'):
                    content_dir = root
                    break
            if content_dir:
                break

        if not content_dir:
            raise Exception("Failed to generate chart (no SM file found)")

        # Create ZIP
        zip_fp = os.path.join(tempfile.gettempdir(), f"{song_id}.zip")
        with zipfile.ZipFile(zip_fp, 'w', zipfile.ZIP_DEFLATED) as z:
            for fn in os.listdir(content_dir):
                z.write(
                    os.path.join(content_dir, fn),
                    os.path.join(f"{song_artist} - {song_title}", fn) # Simple folder name in zip
                )

        return send_file(
            zip_fp,
            as_attachment=True,
            download_name=f"{song_artist} - {song_title}.zip"
        )

    except Exception as e:
        print(f"Error: {e}")
        return f'Error: {str(e)}', 500
    finally:
        shutil.rmtree(out_dir)

@app.after_request
def add_header(r):
    r.headers['Access-Control-Allow-Origin'] = '*'
    r.headers['Access-Control-Allow-Headers'] = '*'
    r.headers['Access-Control-Allow-Methods'] = '*'
    r.headers['Access-Control-Expose-Headers'] = '*'
    return r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_dir', type=str, required=True, help='Path to DDC v2 models')
    parser.add_argument('--ffr_dir', type=str, help='Path to FFR models')
    parser.add_argument('--google_key', type=str, help='Google API Key')
    parser.add_argument('--cx', type=str, help='Google Custom Search Engine ID')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--max_file_size', type=int, default=1024 * 1024 * 50) # 50MB

    global ARGS
    ARGS = parser.parse_args()

    app.config['MAX_CONTENT_LENGTH'] = ARGS.max_file_size
    app.run(host='0.0.0.0', port=ARGS.port)
