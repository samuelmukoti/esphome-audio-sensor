#!/usr/bin/env python3
"""
Ensemble Beep Detector - Combines Three Detection Methods

Architecture:
    Audio Input (16kHz mono)
        │
        ├─> YAMNet Classifier (60% weight)
        │   └─> Detect classes: Beep (67), Alarm (87), Buzzer, Siren
        │
        ├─> Frequency Peak Detector (20% weight)
        │   └─> Analyze 1-6kHz range for beep signature
        │   └─> Use FFT/spectrogram to find dominant peaks
        │
        └─> Energy Detector (20% weight)
            └─> Detect sharp energy peaks (impulse response)
            └─> Time-domain analysis

        ↓
    Weighted Voting (threshold > 0.35)
        ↓
    BEEP / NO-BEEP Decision

Usage:
    from ensemble_detector import EnsembleBeepDetector

    detector = EnsembleBeepDetector(
        yamnet_weight=0.4,
        frequency_weight=0.3,
        energy_weight=0.3,
        threshold=0.5
    )

    # Process audio
    score, components = detector.detect_beep(audio_data, sample_rate)

    # Check if beep detected
    is_beep = score > detector.threshold

Configuration:
    - YAMNet weight: 0.6 (primary detector)
    - Frequency weight: 0.2 (supports detection with 1-6kHz range)
    - Energy weight: 0.2 (impulse response analysis)
    - Threshold: 0.35 (optimized for beep sensitivity)
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Dict, Optional

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa

# Try to import TensorFlow and TensorFlow Hub for YAMNet
TF_AVAILABLE = False
TF_HUB_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    try:
        import tensorflow_hub as hub
        TF_HUB_AVAILABLE = True
        print("[INFO] Using TensorFlow Hub for YAMNet inference")
    except ImportError:
        print("[WARNING] TensorFlow Hub not available. YAMNet disabled.")
except ImportError:
    print("[WARNING] TensorFlow not available. YAMNet component disabled.")


class YAMNetComponent:
    """YAMNet-based audio classification component using TensorFlow Hub."""

    # YAMNet class indices for beep-like sounds
    BEEP_CLASSES = {
        67: "Beep, bleep",
        87: "Alarm",
        68: "Sine wave",
        132: "Siren",
        81: "Buzzer",
    }

    def __init__(self, model_url: Optional[str] = None):
        """Initialize YAMNet component.

        Args:
            model_url: TensorFlow Hub URL for YAMNet. If None, uses default.
        """
        self.model = None
        self.enabled = False

        if not TF_AVAILABLE or not TF_HUB_AVAILABLE:
            warnings.warn("TensorFlow or TensorFlow Hub not available. YAMNet disabled.")
            return

        if model_url is None:
            model_url = 'https://tfhub.dev/google/yamnet/1'

        try:
            # Load YAMNet model from TensorFlow Hub
            print(f"[INFO] Loading YAMNet from TensorFlow Hub...")
            self.model = hub.load(model_url)
            self.enabled = True
            print(f"[INFO] YAMNet loaded successfully")

        except Exception as e:
            warnings.warn(f"Failed to load YAMNet model: {e}")
            self.enabled = False

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Run YAMNet classification and return beep probability.

        Args:
            audio_data: Audio samples (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            Float score 0.0-1.0 indicating beep probability
        """
        if not self.enabled:
            return 0.0

        try:
            # Resample to 16kHz if needed (YAMNet expects 16kHz)
            if sample_rate != 16000:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=sample_rate,
                    target_sr=16000
                )

            # Normalize audio to [-1, 1]
            audio_data = audio_data.astype(np.float32)
            if len(audio_data) > 0 and np.abs(audio_data).max() > 0:
                audio_data = audio_data / np.abs(audio_data).max()

            # Run YAMNet inference
            # YAMNet returns (scores, embeddings, spectrogram)
            scores, embeddings, spectrogram = self.model(audio_data)

            # scores shape: [num_chunks, 521 classes]
            # Get max confidence for beep-related classes across all chunks
            beep_scores = []
            for chunk_scores in scores.numpy():
                max_beep_score = max(
                    chunk_scores[class_idx]
                    for class_idx in self.BEEP_CLASSES.keys()
                )
                beep_scores.append(max_beep_score)

            # Return maximum beep score across all chunks
            return float(np.max(beep_scores)) if beep_scores else 0.0

        except Exception as e:
            warnings.warn(f"YAMNet inference failed: {e}")
            return 0.0


class FrequencyPeakDetector:
    """Frequency-domain beep detection using FFT analysis."""

    def __init__(
        self,
        target_freq_min: float = 3000.0,
        target_freq_max: float = 4000.0,
        prominence_threshold: float = 0.3
    ):
        """Initialize frequency peak detector.

        Args:
            target_freq_min: Minimum frequency for beep detection (Hz)
            target_freq_max: Maximum frequency for beep detection (Hz)
            prominence_threshold: Minimum peak prominence (0.0-1.0)
        """
        self.target_freq_min = target_freq_min
        self.target_freq_max = target_freq_max
        self.prominence_threshold = prominence_threshold

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Detect beep-like frequency peaks.

        Args:
            audio_data: Audio samples (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            Float score 0.0-1.0 indicating beep probability
        """
        try:
            # Compute FFT
            n = len(audio_data)
            fft_values = np.abs(fft(audio_data))
            freqs = fftfreq(n, 1/sample_rate)

            # Only look at positive frequencies
            positive_freqs_mask = freqs >= 0
            freqs = freqs[positive_freqs_mask]
            fft_values = fft_values[positive_freqs_mask]

            # Normalize FFT values
            if fft_values.max() > 0:
                fft_values = fft_values / fft_values.max()

            # Find target frequency range
            freq_mask = (freqs >= self.target_freq_min) & (freqs <= self.target_freq_max)
            target_band_power = np.mean(fft_values[freq_mask]) if freq_mask.any() else 0.0

            # Find peaks in the target band
            target_fft = fft_values[freq_mask]
            if len(target_fft) > 10:  # Need enough samples to find peaks
                peaks, properties = signal.find_peaks(
                    target_fft,
                    prominence=self.prominence_threshold,
                    height=0.2  # Minimum peak height
                )

                if len(peaks) > 0:
                    # Score based on:
                    # 1. Peak prominence
                    # 2. Overall energy in target band
                    # 3. Number of peaks (expect 1-2 for pure beep)
                    max_prominence = np.max(properties['prominences']) if 'prominences' in properties else 0.0

                    # Penalize too many peaks (noise-like)
                    peak_penalty = 1.0 if len(peaks) <= 3 else 0.5

                    score = (max_prominence * 0.5 + target_band_power * 0.5) * peak_penalty
                    return float(np.clip(score, 0.0, 1.0))

            # No significant peaks found, use band power as fallback
            return float(np.clip(target_band_power, 0.0, 1.0))

        except Exception as e:
            warnings.warn(f"Frequency peak detection failed: {e}")
            return 0.0


class EnergyDetector:
    """Energy-based beep detection using time-domain analysis."""

    def __init__(
        self,
        window_ms: float = 25.0,
        peak_threshold: float = 3.0,
        impulse_ratio_threshold: float = 2.5
    ):
        """Initialize energy detector.

        Args:
            window_ms: Window size for RMS calculation (milliseconds)
            peak_threshold: Minimum peak-to-median ratio for detection
            impulse_ratio_threshold: Minimum ratio between peak and surrounding energy
        """
        self.window_ms = window_ms
        self.peak_threshold = peak_threshold
        self.impulse_ratio_threshold = impulse_ratio_threshold

    def detect(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Detect sharp energy peaks (impulse-like characteristics).

        Args:
            audio_data: Audio samples (1D numpy array)
            sample_rate: Sample rate in Hz

        Returns:
            Float score 0.0-1.0 indicating beep probability
        """
        try:
            # Calculate window size in samples
            window_samples = int(self.window_ms * sample_rate / 1000.0)

            if len(audio_data) < window_samples:
                return 0.0

            # Compute RMS energy over sliding windows
            hop_length = window_samples // 2  # 50% overlap
            rms_energy = librosa.feature.rms(
                y=audio_data,
                frame_length=window_samples,
                hop_length=hop_length
            )[0]

            if len(rms_energy) < 3:
                return 0.0

            # Normalize RMS values
            if rms_energy.max() > 0:
                rms_energy = rms_energy / rms_energy.max()

            # Find peaks in RMS energy
            peaks, properties = signal.find_peaks(
                rms_energy,
                prominence=0.2,
                height=0.3
            )

            if len(peaks) == 0:
                return 0.0

            # Analyze peak characteristics
            median_energy = np.median(rms_energy)
            max_peak_energy = np.max(rms_energy[peaks])

            # Peak-to-median ratio (beeps have sharp peaks)
            peak_ratio = max_peak_energy / (median_energy + 1e-6)

            # Analyze impulse characteristics (sudden rise, then fall)
            impulse_scores = []
            for peak_idx in peaks:
                # Get surrounding context
                context_start = max(0, peak_idx - 2)
                context_end = min(len(rms_energy), peak_idx + 3)
                context = rms_energy[context_start:context_end]

                if len(context) >= 3:
                    # Energy before and after peak
                    before_energy = np.mean(context[:peak_idx - context_start])
                    after_energy = np.mean(context[peak_idx - context_start + 1:])
                    avg_surrounding = (before_energy + after_energy) / 2.0

                    # Impulse ratio
                    impulse_ratio = rms_energy[peak_idx] / (avg_surrounding + 1e-6)
                    impulse_scores.append(impulse_ratio)

            max_impulse_ratio = np.max(impulse_scores) if impulse_scores else 0.0

            # Combine metrics
            # 1. Peak-to-median ratio (sharp peaks)
            # 2. Impulse ratio (sudden energy change)
            # 3. Number of peaks (penalize too many = noise)
            peak_score = np.clip((peak_ratio - 1.0) / (self.peak_threshold - 1.0), 0.0, 1.0)
            impulse_score = np.clip(
                (max_impulse_ratio - 1.0) / (self.impulse_ratio_threshold - 1.0),
                0.0,
                1.0
            )

            # Penalize too many peaks (noise-like)
            peak_penalty = 1.0 if len(peaks) <= 5 else 0.5

            final_score = (peak_score * 0.5 + impulse_score * 0.5) * peak_penalty
            return float(np.clip(final_score, 0.0, 1.0))

        except Exception as e:
            warnings.warn(f"Energy detection failed: {e}")
            return 0.0


class EnsembleBeepDetector:
    """Ensemble beep detector combining YAMNet, frequency, and energy detection."""

    def __init__(
        self,
        yamnet_weight: float = 0.6,
        frequency_weight: float = 0.2,
        energy_weight: float = 0.2,
        threshold: float = 0.35,
        yamnet_model_url: Optional[str] = None
    ):
        """Initialize ensemble detector.

        Args:
            yamnet_weight: Weight for YAMNet component (0.0-1.0)
            frequency_weight: Weight for frequency detector (0.0-1.0)
            energy_weight: Weight for energy detector (0.0-1.0)
            threshold: Detection threshold for final ensemble score
            yamnet_model_url: TensorFlow Hub URL for YAMNet (optional)
        """
        # Normalize weights to sum to 1.0
        total_weight = yamnet_weight + frequency_weight + energy_weight
        self.yamnet_weight = yamnet_weight / total_weight
        self.frequency_weight = frequency_weight / total_weight
        self.energy_weight = energy_weight / total_weight
        self.threshold = threshold

        # Initialize components
        self.yamnet = YAMNetComponent(model_url=yamnet_model_url)
        self.frequency_detector = FrequencyPeakDetector(
            target_freq_min=1000.0,
            target_freq_max=6000.0
        )
        self.energy_detector = EnergyDetector()

        print(f"[INFO] Ensemble weights: YAMNet={self.yamnet_weight:.2f}, "
              f"Frequency={self.frequency_weight:.2f}, "
              f"Energy={self.energy_weight:.2f}")
        print(f"[INFO] Detection threshold: {self.threshold:.2f}")

    def detect_beep(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> Tuple[float, Dict[str, float]]:
        """Detect beep in audio using ensemble of three methods.

        Args:
            audio_data: Audio samples (1D numpy array, mono)
            sample_rate: Sample rate in Hz

        Returns:
            Tuple of (ensemble_score, component_scores_dict)
            - ensemble_score: Weighted ensemble score (0.0-1.0)
            - component_scores_dict: Individual component scores
        """
        # Get scores from each component
        yamnet_score = self.yamnet.detect(audio_data, sample_rate)
        frequency_score = self.frequency_detector.detect(audio_data, sample_rate)
        energy_score = self.energy_detector.detect(audio_data, sample_rate)

        # Compute weighted ensemble score
        ensemble_score = (
            yamnet_score * self.yamnet_weight +
            frequency_score * self.frequency_weight +
            energy_score * self.energy_weight
        )

        # Component breakdown
        components = {
            'yamnet': yamnet_score,
            'frequency': frequency_score,
            'energy': energy_score,
            'ensemble': ensemble_score,
            'is_beep': ensemble_score > self.threshold
        }

        return ensemble_score, components

    def get_config(self) -> Dict:
        """Get detector configuration.

        Returns:
            Dictionary with detector configuration
        """
        return {
            'weights': {
                'yamnet': self.yamnet_weight,
                'frequency': self.frequency_weight,
                'energy': self.energy_weight
            },
            'threshold': self.threshold,
            'yamnet_enabled': self.yamnet.enabled,
            'components': {
                'yamnet': {
                    'enabled': self.yamnet.enabled,
                    'classes': list(YAMNetComponent.BEEP_CLASSES.keys())
                },
                'frequency': {
                    'freq_min': self.frequency_detector.target_freq_min,
                    'freq_max': self.frequency_detector.target_freq_max,
                    'prominence_threshold': self.frequency_detector.prominence_threshold
                },
                'energy': {
                    'window_ms': self.energy_detector.window_ms,
                    'peak_threshold': self.energy_detector.peak_threshold,
                    'impulse_ratio_threshold': self.energy_detector.impulse_ratio_threshold
                }
            }
        }


if __name__ == "__main__":
    """Quick test of ensemble detector."""
    import soundfile as sf

    # Create detector
    detector = EnsembleBeepDetector()

    # Print configuration
    config = detector.get_config()
    print("\n[CONFIG]")
    print(f"Weights: {config['weights']}")
    print(f"Threshold: {config['threshold']}")
    print(f"YAMNet enabled: {config['yamnet_enabled']}")

    # Test on a sample file if available
    test_file = Path(__file__).parent.parent / "water_heater_beeping_error_sound.m4a"
    if test_file.exists():
        print(f"\n[TEST] Testing on {test_file.name}")
        try:
            # Load audio
            audio, sr = librosa.load(test_file, sr=16000, mono=True)

            # Detect
            score, components = detector.detect_beep(audio, sr)

            print(f"\n[RESULTS]")
            print(f"YAMNet score:    {components['yamnet']:.4f}")
            print(f"Frequency score: {components['frequency']:.4f}")
            print(f"Energy score:    {components['energy']:.4f}")
            print(f"Ensemble score:  {components['ensemble']:.4f}")
            print(f"Is beep:         {components['is_beep']}")

        except Exception as e:
            print(f"Test failed: {e}")
