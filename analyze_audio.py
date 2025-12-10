#!/usr/bin/env python3
"""
Audio Analysis Script for ESPHome Beeping Sensor
Analyzes beep patterns, frequencies, and detection parameters
"""

import wave
import numpy as np
import json
from scipy import signal
from scipy.fft import fft, fftfreq

def analyze_audio(filename):
    """Comprehensive audio analysis for beep detection"""

    # Read WAV file
    with wave.open(filename, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        n_channels = wav_file.getnchannels()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate

        # Read audio data
        audio_data = wav_file.readframes(n_frames)
        audio_array = np.frombuffer(audio_data, dtype=np.int16)

        # Convert to float and normalize
        audio_normalized = audio_array.astype(np.float32) / 32768.0

    print("=" * 80)
    print("AUDIO FILE CHARACTERISTICS")
    print("=" * 80)
    print(f"File: {filename}")
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Channels: {n_channels}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total Samples: {n_frames}")
    print(f"Bit Depth: 16-bit (from WAV)")
    print()

    # Amplitude analysis
    print("=" * 80)
    print("AMPLITUDE ANALYSIS")
    print("=" * 80)
    max_amplitude = np.max(np.abs(audio_normalized))
    mean_amplitude = np.mean(np.abs(audio_normalized))
    rms_amplitude = np.sqrt(np.mean(audio_normalized**2))

    print(f"Peak Amplitude: {max_amplitude:.4f} ({20*np.log10(max_amplitude):.2f} dB)")
    print(f"Mean Amplitude: {mean_amplitude:.4f}")
    print(f"RMS Amplitude: {rms_amplitude:.4f}")
    print()

    # Frequency analysis using FFT
    print("=" * 80)
    print("FREQUENCY SPECTRUM ANALYSIS")
    print("=" * 80)

    # Perform FFT on entire signal
    fft_data = fft(audio_normalized)
    freqs = fftfreq(len(audio_normalized), 1/sample_rate)

    # Only look at positive frequencies
    positive_freqs = freqs[:len(freqs)//2]
    magnitude = np.abs(fft_data[:len(fft_data)//2])

    # Find dominant frequencies (top 10)
    top_indices = np.argsort(magnitude)[-20:][::-1]

    print("Top 10 Dominant Frequencies:")
    for i, idx in enumerate(top_indices[:10]):
        freq = positive_freqs[idx]
        mag = magnitude[idx]
        if freq > 20:  # Ignore DC and very low frequencies
            print(f"  {i+1}. {freq:.1f} Hz - Magnitude: {mag:.0f}")

    # Find primary frequency band
    freq_range = (200, 8000)  # Typical beep range
    freq_mask = (positive_freqs >= freq_range[0]) & (positive_freqs <= freq_range[1])
    primary_freq_idx = np.argmax(magnitude[freq_mask])
    primary_freq = positive_freqs[freq_mask][primary_freq_idx]

    print(f"\nPrimary Frequency (200-8000 Hz): {primary_freq:.1f} Hz")
    print()

    # Beep detection - find energy spikes
    print("=" * 80)
    print("BEEP PATTERN ANALYSIS")
    print("=" * 80)

    # Calculate short-term energy (100ms windows)
    window_size = int(0.1 * sample_rate)  # 100ms windows
    hop_size = int(0.02 * sample_rate)    # 20ms hop

    energy = []
    times = []

    for i in range(0, len(audio_normalized) - window_size, hop_size):
        window = audio_normalized[i:i+window_size]
        window_energy = np.sqrt(np.mean(window**2))
        energy.append(window_energy)
        times.append(i / sample_rate)

    energy = np.array(energy)
    times = np.array(times)

    # Find beeps using energy threshold
    energy_threshold = np.mean(energy) + 2 * np.std(energy)

    # Detect beep events
    beep_active = energy > energy_threshold

    # Find beep start/end times
    beep_starts = []
    beep_ends = []
    in_beep = False

    for i in range(len(beep_active)):
        if beep_active[i] and not in_beep:
            beep_starts.append(times[i])
            in_beep = True
        elif not beep_active[i] and in_beep:
            beep_ends.append(times[i])
            in_beep = False

    if in_beep:
        beep_ends.append(times[-1])

    # Calculate beep characteristics
    beep_durations = []
    beep_intervals = []

    print(f"Energy Threshold: {energy_threshold:.4f}")
    print(f"Number of Beeps Detected: {len(beep_starts)}")
    print()

    if len(beep_starts) > 0:
        print("Detected Beep Events:")
        for i, (start, end) in enumerate(zip(beep_starts, beep_ends)):
            duration = end - start
            beep_durations.append(duration)
            print(f"  Beep {i+1}: {start:.2f}s - {end:.2f}s (duration: {duration:.3f}s)")

            if i > 0:
                interval = start - beep_ends[i-1]
                beep_intervals.append(interval)

        print()

        if len(beep_durations) > 0:
            print(f"Average Beep Duration: {np.mean(beep_durations):.3f}s")
            print(f"Min Beep Duration: {np.min(beep_durations):.3f}s")
            print(f"Max Beep Duration: {np.max(beep_durations):.3f}s")

        if len(beep_intervals) > 0:
            print(f"Average Beep Interval: {np.mean(beep_intervals):.3f}s")
            print(f"Min Beep Interval: {np.min(beep_intervals):.3f}s")
            print(f"Max Beep Interval: {np.max(beep_intervals):.3f}s")

    print()

    # Noise floor analysis
    print("=" * 80)
    print("NOISE CHARACTERISTICS")
    print("=" * 80)

    # Estimate noise floor (periods below threshold)
    noise_segments = energy[~beep_active]
    if len(noise_segments) > 0:
        noise_floor = np.mean(noise_segments)
        noise_std = np.std(noise_segments)
        print(f"Noise Floor (RMS): {noise_floor:.4f}")
        print(f"Noise Std Dev: {noise_std:.4f}")

        if len(beep_starts) > 0:
            signal_level = np.mean(energy[beep_active])
            snr = 20 * np.log10(signal_level / noise_floor)
            print(f"Signal Level (RMS): {signal_level:.4f}")
            print(f"Signal-to-Noise Ratio: {snr:.2f} dB")

    print()

    # Spectrogram analysis for time-frequency patterns
    print("=" * 80)
    print("TIME-FREQUENCY ANALYSIS")
    print("=" * 80)

    # Compute spectrogram
    f, t, Sxx = signal.spectrogram(audio_normalized, sample_rate, nperseg=1024)

    # Find frequency bands with most energy
    freq_energy = np.sum(Sxx, axis=1)

    # Focus on audible beep range
    beep_freq_mask = (f >= 500) & (f <= 5000)
    beep_freq_energy = freq_energy[beep_freq_mask]
    beep_freqs = f[beep_freq_mask]

    top_freq_indices = np.argsort(beep_freq_energy)[-5:][::-1]

    print("Frequency Bands with Highest Energy (500-5000 Hz):")
    for i, idx in enumerate(top_freq_indices):
        freq = beep_freqs[idx]
        energy_val = beep_freq_energy[idx]
        print(f"  {i+1}. {freq:.1f} Hz - Energy: {energy_val:.0f}")

    print()

    # Recommendations
    print("=" * 80)
    print("DETECTION PARAMETER RECOMMENDATIONS")
    print("=" * 80)

    recommendations = {
        "audio_characteristics": {
            "sample_rate": sample_rate,
            "duration": float(duration),
            "channels": n_channels,
            "peak_amplitude": float(max_amplitude),
            "rms_amplitude": float(rms_amplitude)
        },
        "frequency_analysis": {
            "primary_frequency_hz": float(primary_freq),
            "recommended_bandpass_range": [float(primary_freq * 0.7), float(primary_freq * 1.3)],
            "dominant_frequencies": [float(positive_freqs[idx]) for idx in top_indices[:5] if positive_freqs[idx] > 20]
        },
        "beep_pattern": {
            "num_beeps": len(beep_starts),
            "avg_beep_duration_s": float(np.mean(beep_durations)) if beep_durations else 0,
            "avg_beep_interval_s": float(np.mean(beep_intervals)) if beep_intervals else 0,
            "min_beep_duration_s": float(np.min(beep_durations)) if beep_durations else 0
        },
        "noise_characteristics": {
            "noise_floor": float(noise_floor) if len(noise_segments) > 0 else 0,
            "snr_db": float(snr) if len(beep_starts) > 0 and len(noise_segments) > 0 else 0
        },
        "detection_parameters": {
            "amplitude_threshold": float(energy_threshold * 0.8),  # 80% of detected threshold
            "min_duration_ms": float(np.min(beep_durations) * 1000 * 0.8) if beep_durations else 100,
            "frequency_range_hz": [500, 4000],
            "bandpass_center_hz": float(primary_freq),
            "bandpass_width_hz": 500
        }
    }

    print("\n1. FREQUENCY-BASED DETECTION:")
    print(f"   - Center Frequency: {primary_freq:.1f} Hz")
    print(f"   - Bandpass Filter Range: {primary_freq*0.7:.1f} - {primary_freq*1.3:.1f} Hz")
    print(f"   - Recommended: {recommendations['detection_parameters']['frequency_range_hz'][0]} - {recommendations['detection_parameters']['frequency_range_hz'][1]} Hz")

    print("\n2. AMPLITUDE DETECTION:")
    print(f"   - Minimum Threshold: {energy_threshold*0.8:.4f} (80% of detected)")
    print(f"   - RMS-based threshold recommended")

    print("\n3. DURATION FILTERING:")
    if beep_durations:
        print(f"   - Minimum Duration: {np.min(beep_durations)*800:.0f} ms (80% of shortest beep)")
        print(f"   - Typical Duration: {np.mean(beep_durations)*1000:.0f} ms")

    print("\n4. PATTERN MATCHING:")
    if len(beep_intervals) > 1:
        print(f"   - Beep Interval: ~{np.mean(beep_intervals):.2f}s")
        print(f"   - Pattern: {len(beep_starts)} beeps in {duration:.1f}s")
        print(f"   - Regular pattern: {'Yes' if np.std(beep_intervals)/np.mean(beep_intervals) < 0.2 else 'No'}")

    print("\n5. RECOMMENDED APPROACH:")

    if len(noise_segments) > 0 and len(beep_starts) > 0:
        snr_db = 20 * np.log10(np.mean(energy[beep_active]) / noise_floor)

        if snr_db > 15:
            print("   ✓ SIMPLE FREQUENCY-BASED DETECTION")
            print("     Reason: High SNR, clear frequency signature")
            print("     Implementation: Bandpass filter + RMS threshold")
        elif snr_db > 8:
            print("   ✓ FREQUENCY + PATTERN MATCHING")
            print("     Reason: Moderate SNR, may need pattern confirmation")
            print("     Implementation: Bandpass + RMS + duration check")
        else:
            print("   ✓ ML-BASED CLASSIFICATION")
            print("     Reason: Low SNR, complex background noise")
            print("     Implementation: Edge Impulse or similar")

    print("\n6. PREPROCESSING STEPS:")
    print("   1. Bandpass filter (500-4000 Hz) to remove noise")
    print("   2. RMS calculation over sliding window (50-100ms)")
    print("   3. Threshold detection with hysteresis")
    print("   4. Duration validation (minimum duration check)")
    print("   5. Optional: Pattern matching for confirmation")

    print()
    print("=" * 80)

    # Save recommendations to JSON
    with open('/Users/sam/tmp/esphome-audio-sensor/audio_analysis_report.json', 'w') as f:
        json.dump(recommendations, f, indent=2)

    print("Analysis saved to: audio_analysis_report.json")
    print("=" * 80)

    return recommendations

if __name__ == "__main__":
    analyze_audio("/Users/sam/tmp/esphome-audio-sensor/analysis_temp.wav")
