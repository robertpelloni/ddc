from pydub import AudioSegment

try:
    # Explicitly point to the ffmpeg binary
    AudioSegment.converter = "/usr/bin/ffmpeg"
    
    # Attempt to load the synthetic MP3 file
    song = AudioSegment.from_file("data/raw/ddr_synthetic/My Synthetic Pack/My Synthetic Song/My Synthetic Song.mp3")
    
    print("Successfully loaded the audio file.")
    
except Exception as e:
    print(f"An error occurred: {e}")
