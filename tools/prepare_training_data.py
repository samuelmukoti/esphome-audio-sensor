#!/usr/bin/env python3
"""
Prepare Training Data for Beep Detection Neural Network

This script:
1. Loads audio file (M4A, WAV, etc.)
2. Converts to 16kHz mono (matching ESP32)
3. Analyzes audio to find beep events
4. Extracts MFCC features for training
5. Saves training data as numpy arrays

Usage:
    python prepare_training_data.py ../water_heater_beeping_error_sound.m4a
"""

import argparse
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# Constants matching ESP32 target
TARGET_SAMPLE_RATE = 16000
WINDOW_DURATION_MS = 500  # 500ms windows for classification
HOP_DURATION_MS = 10  # 10ms hop for MFCC
N_MFCC = 20  # Number of MFCC coefficients


def load_and_convert_audio(audio_path: str) -> tuple[np.ndarray, int]:
    """Load audio file and convert to 16kHz mono."""
    print(f"Loading audio: {audio_path}")

    # Load with librosa (handles M4A, WAV, etc.)
    y, sr = librosa.load(audio_path, sr=TARGET_SAMPLE_RATE, mono=True)

    print(f"  Original sample rate: detected, converted to {TARGET_SAMPLE_RATE} Hz")
    print(f"  Duration: {len(y) / sr:.2f} seconds")
    print(f"  Samples: {len(y)}")

    return y, sr


def find_beep_events(y: np.ndarray, sr: int,
                     energy_threshold: float = 0.02,
                     min_beep_duration_ms: float = 50,
                     min_gap_ms: float = 100) -> list[tuple[float, float]]:
    """
    Find beep events in audio using energy analysis.

    Returns list of (start_time, end_time) tuples in seconds.
    """
    print("\nAnalyzing audio for beep events...")

    # Calculate short-time energy
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop

    # RMS energy per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Find frames above threshold
    is_active = rms > energy_threshold

    # Convert to continuous regions
    events = []
    in_event = False
    event_start = 0

    for i, (t, active) in enumerate(zip(times, is_active)):
        if active and not in_event:
            # Start of event
            in_event = True
            event_start = t
        elif not active and in_event:
            # End of event
            in_event = False
            event_end = t
            duration_ms = (event_end - event_start) * 1000

            if duration_ms >= min_beep_duration_ms:
                events.append((event_start, event_end))

    # Handle event at end of file
    if in_event:
        event_end = times[-1]
        duration_ms = (event_end - event_start) * 1000
        if duration_ms >= min_beep_duration_ms:
            events.append((event_start, event_end))

    # Merge events that are too close together
    merged_events = []
    for start, end in events:
        if merged_events and (start - merged_events[-1][1]) * 1000 < min_gap_ms:
            # Merge with previous event
            merged_events[-1] = (merged_events[-1][0], end)
        else:
            merged_events.append((start, end))

    print(f"  Found {len(merged_events)} beep events")
    for i, (start, end) in enumerate(merged_events):
        print(f"    Event {i+1}: {start:.3f}s - {end:.3f}s (duration: {(end-start)*1000:.0f}ms)")

    return merged_events


def extract_mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract MFCC features from audio segment.

    Returns: (n_frames, n_mfcc) array
    """
    hop_length = int(sr * HOP_DURATION_MS / 1000)
    n_fft = 2048

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Transpose to (n_frames, n_mfcc)
    return mfcc.T


def create_training_samples(y: np.ndarray, sr: int,
                           beep_events: list[tuple[float, float]],
                           window_duration_ms: int = WINDOW_DURATION_MS) -> tuple[np.ndarray, np.ndarray]:
    """
    Create positive (beep) and negative (non-beep) training samples.

    Returns: (X, y) where X is features and y is labels (1=beep, 0=no beep)
    """
    window_samples = int(sr * window_duration_ms / 1000)
    hop_samples = window_samples // 2  # 50% overlap

    # Expected MFCC frames per window
    hop_length = int(sr * HOP_DURATION_MS / 1000)
    expected_frames = window_samples // hop_length

    print(f"\nCreating training samples...")
    print(f"  Window: {window_duration_ms}ms ({window_samples} samples)")
    print(f"  Expected MFCC frames per window: {expected_frames}")

    # Convert beep events to sample indices with padding
    # Add padding around short beeps so they're detected in windows
    beep_padding_ms = 150  # Add 150ms padding around each beep
    beep_padding_samples = int(sr * beep_padding_ms / 1000)

    beep_regions = []
    for start, end in beep_events:
        beep_start = max(0, int(start * sr) - beep_padding_samples)
        beep_end = min(len(y), int(end * sr) + beep_padding_samples)
        beep_regions.append((beep_start, beep_end))

    print(f"  Added {beep_padding_ms}ms padding around beeps for better window coverage")

    def is_beep_window(start_sample: int, end_sample: int) -> bool:
        """Check if window contains any beep event (with padding)."""
        for beep_start, beep_end in beep_regions:
            # Check for any overlap (beep is anywhere in window)
            if start_sample < beep_end and end_sample > beep_start:
                return True
        return False

    X_list = []
    y_list = []

    # Slide window across audio
    for start in range(0, len(y) - window_samples, hop_samples):
        end = start + window_samples
        window = y[start:end]

        # Extract MFCC
        mfcc = extract_mfcc_features(window, sr)

        # Pad or truncate to expected size
        if len(mfcc) < expected_frames:
            mfcc = np.pad(mfcc, ((0, expected_frames - len(mfcc)), (0, 0)))
        elif len(mfcc) > expected_frames:
            mfcc = mfcc[:expected_frames]

        # Determine label
        label = 1 if is_beep_window(start, end) else 0

        X_list.append(mfcc)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    n_positive = np.sum(y == 1)
    n_negative = np.sum(y == 0)
    print(f"  Total samples: {len(y)}")
    print(f"  Positive (beep): {n_positive}")
    print(f"  Negative (no beep): {n_negative}")

    return X, y


def augment_data(X: np.ndarray, y: np.ndarray,
                 noise_factor: float = 0.1,
                 augmentation_factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment training data with noise and variations.

    Focus on augmenting the minority class (beeps).
    """
    print(f"\nAugmenting data...")

    X_aug_list = [X]
    y_aug_list = [y]

    # Find positive samples
    positive_mask = y == 1
    X_positive = X[positive_mask]
    y_positive = y[positive_mask]

    # Augment positive samples more heavily
    for i in range(augmentation_factor):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, X_positive.shape)
        X_noisy = X_positive + noise
        X_aug_list.append(X_noisy)
        y_aug_list.append(y_positive)

        # Time shift (roll along time axis)
        shift = np.random.randint(-5, 5)
        X_shifted = np.roll(X_positive, shift, axis=1)
        X_aug_list.append(X_shifted)
        y_aug_list.append(y_positive)

        # Scale (gain variation)
        scale = np.random.uniform(0.8, 1.2)
        X_scaled = X_positive * scale
        X_aug_list.append(X_scaled)
        y_aug_list.append(y_positive)

    X_augmented = np.concatenate(X_aug_list, axis=0)
    y_augmented = np.concatenate(y_aug_list, axis=0)

    # Shuffle
    indices = np.random.permutation(len(y_augmented))
    X_augmented = X_augmented[indices]
    y_augmented = y_augmented[indices]

    n_positive = np.sum(y_augmented == 1)
    n_negative = np.sum(y_augmented == 0)
    print(f"  After augmentation:")
    print(f"  Total samples: {len(y_augmented)}")
    print(f"  Positive (beep): {n_positive}")
    print(f"  Negative (no beep): {n_negative}")

    return X_augmented, y_augmented


def visualize_audio(y: np.ndarray, sr: int,
                    beep_events: list[tuple[float, float]],
                    output_path: str):
    """Create visualization of audio with detected beeps."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Waveform
    times = np.arange(len(y)) / sr
    axes[0].plot(times, y, linewidth=0.5)
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')

    # Mark beep events
    for start, end in beep_events:
        axes[0].axvspan(start, end, alpha=0.3, color='red', label='Beep')

    # Spectrogram
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram')

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[2])
    axes[2].set_title('MFCC')
    axes[2].set_ylabel('MFCC Coefficient')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")
    plt.close()


def save_training_data(X: np.ndarray, y: np.ndarray, output_dir: str):
    """Save training data as numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)

    # Split into train/val/test (80/10/10)
    n_samples = len(y)
    indices = np.random.permutation(n_samples)

    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Save splits
    np.save(os.path.join(output_dir, 'X_train.npy'), X[train_idx])
    np.save(os.path.join(output_dir, 'y_train.npy'), y[train_idx])
    np.save(os.path.join(output_dir, 'X_val.npy'), X[val_idx])
    np.save(os.path.join(output_dir, 'y_val.npy'), y[val_idx])
    np.save(os.path.join(output_dir, 'X_test.npy'), X[test_idx])
    np.save(os.path.join(output_dir, 'y_test.npy'), y[test_idx])

    print(f"\nTraining data saved to: {output_dir}")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Validation: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")
    print(f"  Feature shape: {X.shape[1:]}")


def main():
    parser = argparse.ArgumentParser(description='Prepare training data for beep detection')
    parser.add_argument('audio_file', help='Path to audio file (M4A, WAV, etc.)')
    parser.add_argument('--output-dir', default='models/training_data',
                        help='Output directory for training data')
    parser.add_argument('--energy-threshold', type=float, default=0.02,
                        help='Energy threshold for beep detection')
    parser.add_argument('--no-augment', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization of audio analysis')

    args = parser.parse_args()

    # Load audio
    y, sr = load_and_convert_audio(args.audio_file)

    # Find beep events
    beep_events = find_beep_events(y, sr, energy_threshold=args.energy_threshold)

    if not beep_events:
        print("\nWARNING: No beep events detected! Try adjusting --energy-threshold")
        print("Generating visualization to help identify correct threshold...")
        args.visualize = True

    # Visualize if requested
    if args.visualize:
        viz_path = os.path.join(args.output_dir, 'audio_analysis.png')
        os.makedirs(args.output_dir, exist_ok=True)
        visualize_audio(y, sr, beep_events, viz_path)

    if not beep_events:
        sys.exit(1)

    # Create training samples
    X, y_labels = create_training_samples(y, sr, beep_events)

    # Augment data
    if not args.no_augment:
        X, y_labels = augment_data(X, y_labels)

    # Save training data
    save_training_data(X, y_labels, args.output_dir)

    print("\n" + "="*50)
    print("Data preparation complete!")
    print("="*50)
    print(f"\nNext step: Run train_beep_model.py to train the model")


if __name__ == '__main__':
    main()
