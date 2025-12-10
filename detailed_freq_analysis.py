#!/usr/bin/env python3
"""
Detailed frequency analysis using spectral analysis on beep segments
"""

import wave
import struct
import math

def analyze_beep_frequency(filename):
    """Focus on frequency content of actual beep sounds"""

    # Read WAV file
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_bytes = wav_file.readframes(n_frames)

        # Convert to samples
        samples = []
        for i in range(0, len(audio_bytes), 2):
            sample = struct.unpack('<h', audio_bytes[i:i+2])[0]
            samples.append(sample / 32768.0)

    print("=" * 80)
    print("DETAILED FREQUENCY ANALYSIS")
    print("=" * 80)
    print()

    # Known beep times from previous analysis
    beep_times = [
        (0.06, 0.16),
        (2.12, 2.20),
        (2.36, 2.46),
        (2.56, 2.66),
        (2.80, 2.90),
        (3.70, 3.80),
        (3.92, 4.00),
        (6.16, 6.20),
        (6.38, 6.42),
        (7.80, 7.88)
    ]

    all_zero_crossing_freqs = []

    print("Analyzing individual beeps:")
    print("-" * 80)

    for i, (start_time, end_time) in enumerate(beep_times, 1):
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)

        beep_segment = samples[start_sample:end_sample]

        # Zero crossing rate
        zero_crossings = 0
        for j in range(1, len(beep_segment)):
            if (beep_segment[j-1] >= 0 and beep_segment[j] < 0) or \
               (beep_segment[j-1] < 0 and beep_segment[j] >= 0):
                zero_crossings += 1

        duration = end_time - start_time
        estimated_freq = (zero_crossings / 2) / duration

        all_zero_crossing_freqs.append(estimated_freq)

        # Find peak-to-peak amplitude
        max_val = max(beep_segment)
        min_val = min(beep_segment)
        amplitude = max_val - min_val

        print(f"Beep {i:2d} ({start_time:.2f}s-{end_time:.2f}s):")
        print(f"  Frequency: ~{estimated_freq:5.0f} Hz")
        print(f"  Amplitude: {amplitude:.4f}")
        print(f"  Zero Crossings: {zero_crossings}")

    print()
    print("-" * 80)

    if all_zero_crossing_freqs:
        avg_freq = sum(all_zero_crossing_freqs) / len(all_zero_crossing_freqs)
        min_freq = min(all_zero_crossing_freqs)
        max_freq = max(all_zero_crossing_freqs)

        print(f"Average Beep Frequency: {avg_freq:.0f} Hz")
        print(f"Frequency Range:        {min_freq:.0f} - {max_freq:.0f} Hz")
        print()

        # Determine optimal bandpass filter
        center_freq = avg_freq
        bandwidth = max_freq - min_freq

        # Add some margin
        low_cutoff = max(200, center_freq - bandwidth * 1.5)
        high_cutoff = min(8000, center_freq + bandwidth * 1.5)

        print("RECOMMENDED FREQUENCY FILTER PARAMETERS:")
        print("-" * 80)
        print(f"Filter Type:       Bandpass (Butterworth 2nd-4th order)")
        print(f"Center Frequency:  {center_freq:.0f} Hz")
        print(f"Low Cutoff:        {low_cutoff:.0f} Hz")
        print(f"High Cutoff:       {high_cutoff:.0f} Hz")
        print(f"Bandwidth:         {bandwidth:.0f} Hz")
        print()

        # Categorize beep type
        if 1000 <= avg_freq <= 2000:
            beep_type = "Low-pitch alarm beep"
        elif 2000 <= avg_freq <= 4000:
            beep_type = "Standard alarm beep"
        elif 4000 <= avg_freq <= 6000:
            beep_type = "High-pitch alarm beep"
        else:
            beep_type = "Unusual frequency"

        print(f"Beep Classification:  {beep_type}")
        print()

        # ESPHome filter recommendations
        print("ESPHome FILTER CONFIGURATION:")
        print("-" * 80)
        print("filters:")
        print(f"  - highpass: {low_cutoff:.0f}  # Remove low-frequency noise")
        print(f"  - lowpass: {high_cutoff:.0f}   # Remove high-frequency noise")
        print()

    print("=" * 80)

if __name__ == "__main__":
    analyze_beep_frequency("/Users/sam/tmp/esphome-audio-sensor/analysis_temp.wav")
