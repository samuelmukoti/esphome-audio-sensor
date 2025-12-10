#!/usr/bin/env python3
"""
Simple Audio Analysis Script - No external dependencies except wave
Uses basic signal processing techniques
"""

import wave
import struct
import math
import json

def simple_fft(data):
    """Very basic DFT implementation for frequency analysis"""
    N = len(data)
    # Limit to reasonable size for performance
    if N > 4096:
        data = data[:4096]
        N = 4096

    freqs = []
    magnitudes = []

    # Only calculate first N/2 frequencies (positive frequencies)
    for k in range(N // 2):
        real = 0
        imag = 0
        for n in range(N):
            angle = 2 * math.pi * k * n / N
            real += data[n] * math.cos(angle)
            imag += -data[n] * math.sin(angle)

        magnitude = math.sqrt(real*real + imag*imag)
        magnitudes.append(magnitude)

    return magnitudes

def analyze_audio_simple(filename):
    """Analyze audio file using basic Python"""

    print("=" * 80)
    print("AUDIO ANALYSIS REPORT")
    print("Water Heater Beeping Error Sound")
    print("=" * 80)
    print()

    # Read WAV file
    with wave.open(filename, 'rb') as wav_file:
        # Get basic parameters
        n_channels = wav_file.getnchannels()
        sampwidth = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / sample_rate

        print("FILE CHARACTERISTICS")
        print("-" * 80)
        print(f"Sample Rate:     {sample_rate} Hz")
        print(f"Channels:        {n_channels} (Mono)")
        print(f"Bit Depth:       {sampwidth * 8}-bit")
        print(f"Duration:        {duration:.2f} seconds")
        print(f"Total Samples:   {n_frames:,}")
        print()

        # Read all audio data
        audio_bytes = wav_file.readframes(n_frames)

        # Convert to samples
        if sampwidth == 2:  # 16-bit
            samples = []
            for i in range(0, len(audio_bytes), 2):
                sample = struct.unpack('<h', audio_bytes[i:i+2])[0]
                samples.append(sample / 32768.0)  # Normalize to -1.0 to 1.0
        else:
            print(f"Unsupported bit depth: {sampwidth * 8}")
            return

    # AMPLITUDE ANALYSIS
    print("AMPLITUDE ANALYSIS")
    print("-" * 80)

    abs_samples = [abs(s) for s in samples]
    max_amplitude = max(abs_samples)
    mean_amplitude = sum(abs_samples) / len(abs_samples)

    # Calculate RMS
    squared_sum = sum(s*s for s in samples)
    rms_amplitude = math.sqrt(squared_sum / len(samples))

    # Convert to dB
    max_db = 20 * math.log10(max_amplitude) if max_amplitude > 0 else -100
    rms_db = 20 * math.log10(rms_amplitude) if rms_amplitude > 0 else -100

    print(f"Peak Amplitude:  {max_amplitude:.4f} ({max_db:.2f} dB)")
    print(f"Mean Amplitude:  {mean_amplitude:.4f}")
    print(f"RMS Amplitude:   {rms_amplitude:.4f} ({rms_db:.2f} dB)")
    print()

    # ENERGY ANALYSIS FOR BEEP DETECTION
    print("BEEP PATTERN ANALYSIS")
    print("-" * 80)

    # Calculate energy in windows
    window_duration = 0.1  # 100ms windows
    hop_duration = 0.02    # 20ms hop

    window_size = int(window_duration * sample_rate)
    hop_size = int(hop_duration * sample_rate)

    energies = []
    times = []

    for i in range(0, len(samples) - window_size, hop_size):
        window = samples[i:i+window_size]
        energy = math.sqrt(sum(s*s for s in window) / len(window))
        energies.append(energy)
        times.append(i / sample_rate)

    # Find threshold
    mean_energy = sum(energies) / len(energies)

    # Calculate standard deviation
    variance = sum((e - mean_energy)**2 for e in energies) / len(energies)
    std_energy = math.sqrt(variance)

    threshold = mean_energy + 2 * std_energy

    # Detect beeps
    beep_starts = []
    beep_ends = []
    in_beep = False

    for i, energy in enumerate(energies):
        if energy > threshold and not in_beep:
            beep_starts.append(times[i])
            in_beep = True
        elif energy <= threshold and in_beep:
            beep_ends.append(times[i])
            in_beep = False

    if in_beep:
        beep_ends.append(times[-1])

    num_beeps = len(beep_starts)

    print(f"Energy Threshold: {threshold:.4f}")
    print(f"Beeps Detected:   {num_beeps}")
    print()

    if num_beeps > 0:
        print("Individual Beep Events:")
        beep_durations = []
        beep_intervals = []

        for i, (start, end) in enumerate(zip(beep_starts, beep_ends)):
            duration = end - start
            beep_durations.append(duration)
            print(f"  Beep {i+1}: {start:6.2f}s -> {end:6.2f}s  (duration: {duration:.3f}s)")

            if i > 0:
                interval = start - beep_ends[i-1]
                beep_intervals.append(interval)

        print()

        if beep_durations:
            avg_duration = sum(beep_durations) / len(beep_durations)
            min_duration = min(beep_durations)
            max_duration = max(beep_durations)

            print(f"Avg Beep Duration: {avg_duration:.3f}s ({avg_duration*1000:.0f} ms)")
            print(f"Min Beep Duration: {min_duration:.3f}s ({min_duration*1000:.0f} ms)")
            print(f"Max Beep Duration: {max_duration:.3f}s ({max_duration*1000:.0f} ms)")
            print()

        if beep_intervals:
            avg_interval = sum(beep_intervals) / len(beep_intervals)
            min_interval = min(beep_intervals)
            max_interval = max(beep_intervals)

            # Check if pattern is regular
            if len(beep_intervals) > 1:
                interval_variance = sum((iv - avg_interval)**2 for iv in beep_intervals) / len(beep_intervals)
                interval_std = math.sqrt(interval_variance)
                regularity = interval_std / avg_interval if avg_interval > 0 else 1.0
            else:
                regularity = 0

            print(f"Avg Beep Interval: {avg_interval:.3f}s")
            print(f"Min Beep Interval: {min_interval:.3f}s")
            print(f"Max Beep Interval: {max_interval:.3f}s")
            print(f"Pattern Regular:   {'Yes' if regularity < 0.2 else 'No'} (variance: {regularity:.2%})")
            print()

    # NOISE ANALYSIS
    print("NOISE CHARACTERISTICS")
    print("-" * 80)

    # Separate beep and non-beep regions
    beep_energies = []
    noise_energies = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            beep_energies.append(energy)
        else:
            noise_energies.append(energy)

    if noise_energies:
        noise_floor = sum(noise_energies) / len(noise_energies)
        noise_var = sum((e - noise_floor)**2 for e in noise_energies) / len(noise_energies)
        noise_std = math.sqrt(noise_var)

        print(f"Noise Floor:      {noise_floor:.4f}")
        print(f"Noise Std Dev:    {noise_std:.4f}")

        if beep_energies:
            signal_level = sum(beep_energies) / len(beep_energies)
            snr = 20 * math.log10(signal_level / noise_floor) if noise_floor > 0 else 100

            print(f"Signal Level:     {signal_level:.4f}")
            print(f"SNR (Signal/Noise): {snr:.2f} dB")

        print()

    # FREQUENCY ANALYSIS (Simple)
    print("FREQUENCY ANALYSIS")
    print("-" * 80)
    print("Performing basic frequency analysis...")

    # Analyze a segment with high energy
    if beep_starts:
        # Take first beep for frequency analysis
        start_sample = int(beep_starts[0] * sample_rate)
        end_sample = min(start_sample + 4096, len(samples))
        beep_segment = samples[start_sample:end_sample]

        # Simple frequency detection - zero crossing rate
        zero_crossings = 0
        for i in range(1, len(beep_segment)):
            if (beep_segment[i-1] >= 0 and beep_segment[i] < 0) or \
               (beep_segment[i-1] < 0 and beep_segment[i] >= 0):
                zero_crossings += 1

        # Estimate fundamental frequency from zero crossings
        duration_analyzed = len(beep_segment) / sample_rate
        estimated_freq = (zero_crossings / 2) / duration_analyzed

        print(f"Estimated Frequency: ~{estimated_freq:.0f} Hz (from zero crossings)")
        print()

        # Perform basic DFT on beep segment
        print("Computing frequency spectrum (this may take a moment)...")
        magnitudes = simple_fft(beep_segment)

        # Find peaks in magnitude spectrum
        freq_bin_size = sample_rate / len(beep_segment)

        # Find top 5 frequencies
        indexed_mags = [(i, mag) for i, mag in enumerate(magnitudes) if i > 0]  # Skip DC
        indexed_mags.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 5 Dominant Frequencies:")
        for i, (bin_idx, magnitude) in enumerate(indexed_mags[:5]):
            freq = bin_idx * freq_bin_size
            if 20 < freq < 20000:  # Audible range
                print(f"  {i+1}. {freq:7.1f} Hz - Magnitude: {magnitude:10.0f}")

        # Determine primary frequency range
        primary_freq_idx = indexed_mags[0][0]
        primary_freq = primary_freq_idx * freq_bin_size

        print(f"\nPrimary Frequency: ~{primary_freq:.0f} Hz")
        print()

    # RECOMMENDATIONS
    print("=" * 80)
    print("DETECTION PARAMETER RECOMMENDATIONS")
    print("=" * 80)
    print()

    print("1. AMPLITUDE DETECTION PARAMETERS")
    print("-" * 80)
    print(f"   Recommended Threshold:  {threshold * 0.8:.4f} (80% of detected)")
    print(f"   Detection Method:       RMS energy over sliding window")
    print(f"   Window Size:            100ms")
    print(f"   Hop Size:               20-50ms")
    print()

    print("2. FREQUENCY FILTERING")
    print("-" * 80)
    if beep_starts:
        low_freq = max(300, primary_freq * 0.6)
        high_freq = min(5000, primary_freq * 1.5)
        print(f"   Bandpass Filter Range:  {low_freq:.0f} - {high_freq:.0f} Hz")
        print(f"   Center Frequency:       ~{primary_freq:.0f} Hz")
    else:
        print(f"   Recommended Range:      500 - 4000 Hz (typical beep range)")
    print(f"   Filter Type:            Butterworth or Chebyshev bandpass")
    print()

    print("3. DURATION FILTERING")
    print("-" * 80)
    if beep_durations:
        min_dur_ms = min_duration * 1000 * 0.7
        typical_dur_ms = avg_duration * 1000
        print(f"   Minimum Duration:       {min_dur_ms:.0f} ms")
        print(f"   Typical Duration:       {typical_dur_ms:.0f} ms")
        print(f"   Max Duration:           {max_duration * 1000:.0f} ms")
    else:
        print(f"   Recommended Min:        80-100 ms")
    print()

    print("4. PATTERN CHARACTERISTICS")
    print("-" * 80)
    if beep_intervals:
        print(f"   Beep Pattern:           {num_beeps} beeps in {duration:.1f}s")
        print(f"   Average Interval:       {avg_interval:.2f}s ({avg_interval*1000:.0f} ms)")
        print(f"   Pattern Type:           {'Regular' if regularity < 0.2 else 'Irregular'}")
        if regularity < 0.2:
            print(f"   Pattern Matching:       Can use interval validation")
            print(f"   Expected Interval:      {avg_interval:.2f}s ± {interval_std:.2f}s")
    else:
        print(f"   Pattern:                Single or sparse beeps")
    print()

    print("5. SIGNAL QUALITY ASSESSMENT")
    print("-" * 80)
    if noise_energies and beep_energies:
        print(f"   Signal-to-Noise:        {snr:.2f} dB")

        if snr > 20:
            quality = "EXCELLENT"
            approach = "Simple frequency + amplitude detection"
            confidence = "Very High"
        elif snr > 15:
            quality = "GOOD"
            approach = "Frequency + amplitude + duration validation"
            confidence = "High"
        elif snr > 10:
            quality = "MODERATE"
            approach = "Bandpass filter + pattern matching recommended"
            confidence = "Medium"
        else:
            quality = "POOR"
            approach = "ML-based classification strongly recommended"
            confidence = "Low for simple methods"

        print(f"   Signal Quality:         {quality}")
        print(f"   Detection Confidence:   {confidence}")
        print()

    print("6. RECOMMENDED DETECTION APPROACH")
    print("-" * 80)
    if noise_energies and beep_energies and snr > 15:
        print("   ✓ SIMPLE FREQUENCY-BASED DETECTION VIABLE")
        print()
        print("   Reason: High signal-to-noise ratio and clear frequency signature")
        print()
        print("   Implementation Steps:")
        print("   1. Apply bandpass filter (500-4000 Hz)")
        print("   2. Calculate RMS energy over 100ms windows")
        print("   3. Detect crossings above threshold")
        print("   4. Validate minimum duration (>80ms)")
        if beep_intervals and regularity < 0.2:
            print(f"   5. Optional: Validate interval pattern (~{avg_interval:.2f}s)")
        print()
        print("   ESPHome Implementation: Feasible with native filters")
    elif noise_energies and beep_energies and snr > 8:
        print("   ✓ HYBRID APPROACH RECOMMENDED")
        print()
        print("   Reason: Moderate SNR requires additional validation")
        print()
        print("   Implementation Steps:")
        print("   1. Bandpass filter (500-4000 Hz)")
        print("   2. RMS energy detection")
        print("   3. Duration validation (>80ms)")
        print("   4. Pattern matching or debouncing")
        print()
        print("   ESPHome Implementation: Feasible with careful tuning")
    else:
        print("   ⚠ ML-BASED CLASSIFICATION RECOMMENDED")
        print()
        print("   Reason: Low SNR or complex noise environment")
        print()
        print("   Recommended Approach:")
        print("   1. Use Edge Impulse or similar platform")
        print("   2. Train on multiple beep samples")
        print("   3. Deploy to ESP32 with TensorFlow Lite")
        print()
        print("   ESPHome Implementation: Requires custom component")
    print()

    print("7. PREPROCESSING PIPELINE")
    print("-" * 80)
    print("   Step 1: High-pass filter (>200 Hz) - Remove rumble/DC offset")
    print("   Step 2: Bandpass filter (500-4000 Hz) - Isolate beep frequencies")
    print("   Step 3: RMS calculation (100ms window) - Smooth energy signal")
    print("   Step 4: Threshold detection - Identify potential beeps")
    print("   Step 5: Duration validation - Filter false positives")
    print("   Step 6: Debouncing (50-100ms) - Prevent multiple triggers")
    if beep_intervals and regularity < 0.2:
        print("   Step 7: Pattern matching - Confirm beep sequence")
    print()

    # Save report
    report = {
        "file_info": {
            "filename": filename,
            "sample_rate_hz": sample_rate,
            "duration_s": duration,
            "channels": n_channels,
            "bit_depth": sampwidth * 8
        },
        "amplitude": {
            "peak": max_amplitude,
            "mean": mean_amplitude,
            "rms": rms_amplitude,
            "peak_db": max_db,
            "rms_db": rms_db
        },
        "beep_pattern": {
            "num_beeps": num_beeps,
            "avg_duration_s": sum(beep_durations) / len(beep_durations) if beep_durations else 0,
            "min_duration_s": min(beep_durations) if beep_durations else 0,
            "max_duration_s": max(beep_durations) if beep_durations else 0,
            "avg_interval_s": sum(beep_intervals) / len(beep_intervals) if beep_intervals else 0,
            "pattern_regular": regularity < 0.2 if beep_intervals else False
        },
        "signal_quality": {
            "noise_floor": noise_floor if noise_energies else 0,
            "signal_level": sum(beep_energies) / len(beep_energies) if beep_energies else 0,
            "snr_db": snr if noise_energies and beep_energies else 0,
            "quality": quality if noise_energies and beep_energies else "UNKNOWN"
        },
        "detection_parameters": {
            "amplitude_threshold": threshold * 0.8,
            "window_size_ms": 100,
            "hop_size_ms": 20,
            "min_duration_ms": min_duration * 700 if beep_durations else 80,
            "bandpass_low_hz": low_freq if beep_starts else 500,
            "bandpass_high_hz": high_freq if beep_starts else 4000,
            "primary_frequency_hz": primary_freq if beep_starts else 0
        },
        "recommended_approach": approach if noise_energies and beep_energies else "Requires analysis"
    }

    with open('/Users/sam/tmp/esphome-audio-sensor/audio_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("Analysis complete!")
    print("Report saved to: audio_analysis_report.json")
    print("=" * 80)

if __name__ == "__main__":
    analyze_audio_simple("/Users/sam/tmp/esphome-audio-sensor/analysis_temp.wav")
