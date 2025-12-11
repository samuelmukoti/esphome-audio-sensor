#!/usr/bin/env python3
"""
Audio Analysis Module for Beep Detection

Provides multiple analysis methods:
- Goertzel algorithm (single-frequency energy)
- FFT analysis (full spectrum)
- RMS amplitude
- Peak detection

This module is designed to match and debug the ESP32 implementation.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Results from audio analysis."""

    rms: float
    energy: float  # Goertzel energy at target frequency
    fft_peak_freq: float  # Dominant frequency from FFT
    fft_peak_magnitude: float
    detected: bool
    dc_offset: float


class AudioAnalyzer:
    """Audio analyzer for beep detection debugging."""

    def __init__(
        self,
        sample_rate: int = 16000,
        target_freq: float = 2615.0,
        energy_threshold: float = 0.01,
        rms_threshold: float = 0.01,
        window_size_ms: int = 100,
    ):
        self.sample_rate = sample_rate
        self.target_freq = target_freq
        self.energy_threshold = energy_threshold
        self.rms_threshold = rms_threshold
        self.window_size_ms = window_size_ms

        # Calculate window size in samples
        self.window_size = int(sample_rate * window_size_ms / 1000)

        # Pre-calculate Goertzel coefficients (matching ESP32 implementation)
        self._update_goertzel_coefficients()

        # DC offset tracking (exponential moving average)
        self.dc_offset = 0.0
        self.dc_alpha = 0.001  # Slow adaptation

        # Detection state (for debouncing)
        self.consecutive_detections = 0
        self.debounce_count = 2

        print(f"AudioAnalyzer initialized:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Target frequency: {target_freq} Hz")
        print(f"  Window size: {self.window_size} samples ({window_size_ms} ms)")
        print(f"  Energy threshold: {energy_threshold}")
        print(f"  RMS threshold: {rms_threshold}")
        print(f"  Goertzel coefficient: {self.coeff:.6f}")

    def _update_goertzel_coefficients(self):
        """Calculate Goertzel coefficients for target frequency."""
        # Match ESP32 implementation exactly
        k = 0.5 + (self.window_size * self.target_freq / self.sample_rate)
        omega = (2.0 * np.pi * k) / self.window_size
        self.coeff = 2.0 * np.cos(omega)
        self.sin_val = np.sin(omega)
        self.cos_val = np.cos(omega)

    def set_target_frequency(self, freq: float):
        """Update target frequency and recalculate coefficients."""
        self.target_freq = freq
        self._update_goertzel_coefficients()
        print(f"Target frequency updated to {freq} Hz (coeff={self.coeff:.6f})")

    def remove_dc_offset(self, samples: np.ndarray) -> np.ndarray:
        """Remove DC offset using exponential moving average (matching ESP32)."""
        output = np.zeros_like(samples, dtype=np.float32)

        for i, sample in enumerate(samples):
            # Update DC estimate
            self.dc_offset = self.dc_offset * (1 - self.dc_alpha) + float(sample) * self.dc_alpha
            # Remove DC
            output[i] = float(sample) - self.dc_offset

        return output.astype(np.int16)

    def calculate_rms(self, samples: np.ndarray) -> float:
        """Calculate RMS amplitude (normalized to 0-1 range)."""
        # Normalize to [-1, 1] range (matching ESP32)
        normalized = samples.astype(np.float32) / 32768.0
        return np.sqrt(np.mean(normalized ** 2))

    def calculate_goertzel(self, samples: np.ndarray) -> float:
        """
        Goertzel algorithm for single-frequency energy detection.
        This implementation matches the ESP32 code exactly.
        """
        q0 = 0.0
        q1 = 0.0
        q2 = 0.0

        # Normalize samples to [-1, 1]
        for sample in samples:
            normalized = float(sample) / 32768.0
            q0 = self.coeff * q1 - q2 + normalized
            q2 = q1
            q1 = q0

        # Calculate magnitude squared
        real = q1 - q2 * self.cos_val
        imag = q2 * self.sin_val
        magnitude_squared = real * real + imag * imag

        # Normalize by window size (matching ESP32)
        return magnitude_squared / len(samples)

    def calculate_fft(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate FFT and return frequency bins and magnitudes.
        Useful for visualizing full spectrum.
        """
        # Apply Hanning window to reduce spectral leakage
        windowed = samples * np.hanning(len(samples))

        # FFT
        fft_result = np.fft.rfft(windowed)
        magnitudes = np.abs(fft_result) / len(samples)

        # Frequency bins
        freqs = np.fft.rfftfreq(len(samples), 1 / self.sample_rate)

        return freqs, magnitudes

    def find_peak_frequency(self, samples: np.ndarray) -> tuple[float, float]:
        """Find the dominant frequency in the signal."""
        freqs, magnitudes = self.calculate_fft(samples)

        # Find peak (ignore DC component at index 0)
        peak_idx = np.argmax(magnitudes[1:]) + 1
        peak_freq = freqs[peak_idx]
        peak_mag = magnitudes[peak_idx]

        return peak_freq, peak_mag

    def analyze(self, samples: np.ndarray) -> dict:
        """
        Perform full analysis on audio samples.
        Returns dict with all analysis results.
        """
        # Store original DC offset for reporting
        original_dc = self.dc_offset

        # Remove DC offset (matching ESP32)
        dc_removed = self.remove_dc_offset(samples)

        # Calculate metrics
        rms = self.calculate_rms(dc_removed)
        energy = self.calculate_goertzel(dc_removed)
        peak_freq, peak_mag = self.find_peak_frequency(dc_removed)

        # Detection logic (matching ESP32)
        frequency_match = energy > self.energy_threshold
        amplitude_match = rms > self.rms_threshold
        detected = frequency_match and amplitude_match

        # Update debounce counter
        if detected:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        # Only report detection after debounce threshold
        confirmed_detection = self.consecutive_detections >= self.debounce_count

        return {
            "rms": rms,
            "energy": energy,
            "fft_peak_freq": peak_freq,
            "fft_peak_magnitude": peak_mag,
            "detected": confirmed_detection,
            "dc_offset": original_dc,
            "frequency_match": frequency_match,
            "amplitude_match": amplitude_match,
            "consecutive_detections": self.consecutive_detections,
        }

    def analyze_file(self, filename: str) -> list[dict]:
        """
        Analyze a WAV file and return analysis for each window.
        Useful for offline analysis of recordings.
        """
        import wave

        results = []

        with wave.open(filename, "rb") as wav:
            n_channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()

            print(f"Analyzing: {filename}")
            print(f"  Channels: {n_channels}, Sample width: {sample_width}, Rate: {framerate}")
            print(f"  Frames: {n_frames}, Duration: {n_frames / framerate:.2f}s")

            if framerate != self.sample_rate:
                print(f"  WARNING: File sample rate ({framerate}) differs from analyzer ({self.sample_rate})")

            # Read all frames
            raw_data = wav.readframes(n_frames)

            # Convert to numpy array
            if sample_width == 2:
                samples = np.frombuffer(raw_data, dtype=np.int16)
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")

            # If stereo, take left channel only
            if n_channels == 2:
                samples = samples[::2]

            # Analyze in windows
            for i in range(0, len(samples) - self.window_size, self.window_size // 2):
                window = samples[i : i + self.window_size]
                result = self.analyze(window)
                result["timestamp_ms"] = (i / framerate) * 1000
                results.append(result)

        return results

    def print_analysis_summary(self, results: list[dict]):
        """Print summary of analysis results from a file."""
        if not results:
            print("No analysis results")
            return

        rms_values = [r["rms"] for r in results]
        energy_values = [r["energy"] for r in results]
        detections = sum(1 for r in results if r["detected"])

        print(f"\nAnalysis Summary:")
        print(f"  Windows analyzed: {len(results)}")
        print(f"  RMS: min={min(rms_values):.4f}, max={max(rms_values):.4f}, mean={np.mean(rms_values):.4f}")
        print(f"  Energy: min={min(energy_values):.6f}, max={max(energy_values):.6f}, mean={np.mean(energy_values):.6f}")
        print(f"  Detections: {detections}")

        # Find detection timestamps
        if detections > 0:
            print(f"\n  Detection timestamps:")
            for r in results:
                if r["detected"]:
                    print(f"    {r['timestamp_ms']:.0f}ms: energy={r['energy']:.6f}, rms={r['rms']:.4f}")


def analyze_wav_file(filename: str, target_freq: float = 2615.0):
    """Convenience function to analyze a WAV file."""
    analyzer = AudioAnalyzer(target_freq=target_freq)
    results = analyzer.analyze_file(filename)
    analyzer.print_analysis_summary(results)
    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Analyze provided WAV file
        filename = sys.argv[1]
        target_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 2615.0
        analyze_wav_file(filename, target_freq)
    else:
        print("Usage: python audio_analyzer.py <wav_file> [target_freq]")
        print("\nExample:")
        print("  python audio_analyzer.py recording.wav 2615")
