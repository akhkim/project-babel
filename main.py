import soundcard as sc
import soundfile as sf
import numpy as np
import os
import argparse
from faster_whisper import WhisperModel
from translatepy.translators.google import GoogleTranslate
import time
 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tiny", help="Model to use",
                    choices=["tiny", "base", "small", "medium", "large"])
parser.add_argument("--translation_lang", default='English',
                    help="Which language should we translate into?" , type=str)
parser.add_argument("--energy_threshold", default=0.00005,
                    help="How loud of a sound should we record?" , type=float)
args = parser.parse_args()
if args.model == "large":
    args.model = "distil-large-v2"

gtranslate = GoogleTranslate()

samplerate = 16000  # Increased for better audio quality
chunk_duration = 1
energy_threshold = args.energy_threshold
min_record_duration = 2
output = "output.wav"

model = WhisperModel(args.model, device="auto", compute_type="auto")

def is_loud_enough(data):
    loudness = np.sqrt(np.mean(data**2))
    print(f"Current loudness: {loudness:.4f}")  # Debugging output
    return loudness >= energy_threshold

with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=samplerate) as mic:
    print(f"Recording... Press Ctrl+C to stop. Energy threshold: {energy_threshold}")
    try:
        while True:
            buffer = []
            recording = False
            record_duration = 0
            silent_duration = 0

            while True:
                chunk = mic.record(numframes=samplerate * chunk_duration)
                if is_loud_enough(chunk):
                    if not recording:
                        print("Started recording...")
                        recording = True
                    buffer.append(chunk)
                    record_duration += chunk_duration
                    silent_duration = 0
                elif recording:
                    buffer.append(chunk)
                    record_duration += chunk_duration
                    silent_duration += chunk_duration
                    if silent_duration >= 1.0:  # Stop after 1 second of silence
                        print(f"Stopped recording. Duration: {record_duration:.2f}s")
                        break

            if buffer and record_duration >= min_record_duration:
                full_recording = np.concatenate(buffer, axis=0)
                sf.write(file=output, data=full_recording[:, 0], samplerate=samplerate)
                print(f"Saved recording of length {record_duration:.2f}s to {output}")
                
                print("Transcribing...")
                segments, info = model.transcribe(output, beam_size=5)
                for segment in segments:
                    translated = gtranslate.translate(segment.text, args.translation_lang)
                    print(f"Original: {segment.text}")
                    print(f"Translated: {translated}")
                print("Ready for next recording...")
            else:
                print("Recording too short, discarded.")

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("Recording stopped by user.")
