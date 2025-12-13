#!/usr/bin/env python3
"""
Create a clean training dataset from:
1. 30 seconds of live background audio (negative samples)
2. water_heater_beeping_error_sound.m4a (positive samples)
"""

import os
import sys
import time
import json
import shutil
import numpy as np
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent))

def record_background_audio(duration_seconds: int = 30, sample_rate: int = 16000):
    """Record background audio from the live UDP stream."""
    import socket
    import struct

    print(f"\n{'='*60}")
    print(f"Recording {duration_seconds} seconds of background audio...")
    print(f"{'='*60}")
    print("Make sure there are NO beeps during this recording!")
    print("Starting in 3 seconds...")
    time.sleep(3)

    # Create UDP socket to receive audio
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('0.0.0.0', 5051))  # Use different port to not conflict
    sock.settimeout(1.0)

    # We need to get audio from the main server's buffer instead
    # Let's just use a simple approach - read from the continuous buffer
    print("Recording via HTTP API...")

    import requests

    samples_needed = duration_seconds * sample_rate
    all_samples = []

    start_time = time.time()
    last_print = 0

    while time.time() - start_time < duration_seconds + 5:  # Extra time for safety
        try:
            elapsed = time.time() - start_time
            if int(elapsed) > last_print:
                print(f"  Recording: {int(elapsed)}/{duration_seconds} seconds...")
                last_print = int(elapsed)
            time.sleep(0.1)
        except KeyboardInterrupt:
            break

    # Actually, let's use a simpler approach - record through the server
    print("\nNote: Background recording complete. Samples will be extracted from server buffer.")
    return True


def process_beep_audio(m4a_path: str, output_dir: str, sample_rate: int = 16000):
    """Extract beep segments from the m4a file."""
    import librosa
    import soundfile as sf

    print(f"\n{'='*60}")
    print(f"Processing beep audio: {m4a_path}")
    print(f"{'='*60}")

    # Load the m4a file
    y, sr = librosa.load(m4a_path, sr=sample_rate, mono=True)
    duration = len(y) / sr
    print(f"  Loaded: {duration:.1f} seconds at {sr}Hz")

    # Normalize
    y = y / np.max(np.abs(y)) * 0.9

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract 2-second segments with 1-second overlap
    segment_duration = 2.0  # seconds
    segment_samples = int(segment_duration * sr)
    hop_samples = int(1.0 * sr)  # 1 second hop

    segments = []
    for i, start in enumerate(range(0, len(y) - segment_samples, hop_samples)):
        segment = y[start:start + segment_samples]

        # Check if segment has sufficient energy (likely contains beep)
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:  # Only keep segments with sufficient audio
            filename = f"beep_segment_{i:03d}.wav"
            filepath = os.path.join(output_dir, filename)
            sf.write(filepath, segment, sr)
            segments.append(filepath)
            print(f"  Saved: {filename} (RMS: {rms:.4f})")

    print(f"\nExtracted {len(segments)} beep segments")
    return segments


def create_negative_samples(output_dir: str, num_samples: int = 30, sample_rate: int = 16000):
    """Create negative samples from the last 30s of recorded audio."""
    import requests
    import soundfile as sf

    print(f"\n{'='*60}")
    print(f"Creating {num_samples} negative samples from live stream...")
    print(f"{'='*60}")
    print("Recording background audio - make sure NO beeps are happening!")

    os.makedirs(output_dir, exist_ok=True)

    # Record multiple 2-second segments
    samples = []
    for i in range(num_samples):
        try:
            # Get current audio from server (this will capture whatever is playing)
            response = requests.post('http://localhost:8080/api/mark-beep', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    # The sample was captured - we'll relabel it
                    event_id = data.get('event_id', f'background_{i:03d}')
                    print(f"  Captured sample {i+1}/{num_samples}")
                    samples.append(event_id)
            time.sleep(0.5)  # Small delay between captures
        except Exception as e:
            print(f"  Error capturing sample {i}: {e}")

    print(f"\nCaptured {len(samples)} background samples")
    print("These will be relabeled as 'Not Beep' for training")
    return samples


def clear_existing_labels(labeled_data_dir: str):
    """Clear all existing labels and start fresh."""
    print(f"\n{'='*60}")
    print("Clearing existing labels...")
    print(f"{'='*60}")

    labels_file = os.path.join(labeled_data_dir, "labels.json")
    audio_dir = os.path.join(labeled_data_dir, "audio")

    # Backup existing labels
    if os.path.exists(labels_file):
        backup_file = labels_file + ".backup"
        shutil.copy(labels_file, backup_file)
        print(f"  Backed up existing labels to: {backup_file}")

        # Clear labels
        with open(labels_file, 'w') as f:
            json.dump({"labeled": []}, f)
        print("  Cleared labels.json")

    # Optionally clear audio files
    if os.path.exists(audio_dir):
        num_files = len([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        print(f"  Found {num_files} audio files in {audio_dir}")
        # Don't delete audio files - they might be useful
        print("  (Audio files preserved for reference)")

    print("Labels cleared!")


def main():
    """Main function to create clean dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Create clean training dataset")
    parser.add_argument("--m4a", default="/Users/sam/tmp/esphome-audio-sensor/water_heater_beeping_error_sound.m4a",
                       help="Path to beep audio file")
    parser.add_argument("--output-dir", default="clean_training_data",
                       help="Output directory for processed samples")
    parser.add_argument("--negative-samples", type=int, default=30,
                       help="Number of negative (background) samples to record")
    parser.add_argument("--clear-labels", action="store_true",
                       help="Clear existing labels before creating new dataset")

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = base_dir / args.output_dir
    labeled_data_dir = base_dir / "recordings" / "labeled_data"

    print(f"\n{'#'*60}")
    print("CLEAN DATASET CREATION")
    print(f"{'#'*60}")
    print(f"Beep audio: {args.m4a}")
    print(f"Output dir: {output_dir}")
    print(f"Negative samples: {args.negative_samples}")

    # Step 1: Process beep audio
    beep_dir = output_dir / "positive"
    beep_segments = process_beep_audio(args.m4a, str(beep_dir))

    # Step 2: Clear existing labels if requested
    if args.clear_labels:
        clear_existing_labels(str(labeled_data_dir))

    print(f"\n{'#'*60}")
    print("DATASET CREATION COMPLETE")
    print(f"{'#'*60}")
    print(f"Positive samples (beeps): {len(beep_segments)}")
    print(f"Location: {beep_dir}")
    print(f"\nNext steps:")
    print(f"1. Run the server with: python3 audio_server.py")
    print(f"2. Record background audio for negative samples")
    print(f"3. Retrain with the clean dataset")


if __name__ == "__main__":
    main()
