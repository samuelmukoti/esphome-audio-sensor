#!/usr/bin/env python3
"""
Audio Streaming Server for ESPHome Beep Detector Debugging

Receives UDP audio stream from ESP32 and provides:
- Real-time audio analysis (RMS, Goertzel energy, FFT)
- Neural network beep detection
- WAV file recording
- Detection event logging
- MQTT integration for Home Assistant

Usage:
    # Live UDP stream with NN detection
    python audio_server.py --port 5000 --use-nn

    # Test on audio file
    python audio_server.py --test-file ../water_heater_beeping_error_sound.m4a --use-nn

Requirements:
    pip install numpy scipy paho-mqtt tensorflow librosa
"""

import argparse
import socket
import struct
import time
import wave
import os
from datetime import datetime
from collections import deque
from typing import Optional
from pathlib import Path

import numpy as np

from audio_analyzer import AudioAnalyzer


class NeuralBeepDetector:
    """Neural network-based beep detector using trained CNN model."""

    def __init__(
        self,
        model_path: str = "models/beep_detector.keras",
        sample_rate: int = 16000,
        window_duration_ms: int = 500,
        hop_duration_ms: int = 10,
        n_mfcc: int = 20,
        confidence_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.window_duration_ms = window_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.n_mfcc = n_mfcc
        self.confidence_threshold = confidence_threshold

        # Calculate window size in samples
        self.window_samples = int(sample_rate * window_duration_ms / 1000)
        self.hop_samples = int(sample_rate * hop_duration_ms / 1000)

        # Buffer for accumulating audio
        self.audio_buffer = np.array([], dtype=np.int16)

        # Load model
        self.model = None
        self._load_model(model_path)

        # Detection state for debouncing
        self.consecutive_detections = 0
        self.debounce_count = 2

        print(f"NeuralBeepDetector initialized:")
        print(f"  Model: {model_path}")
        print(f"  Window: {window_duration_ms}ms ({self.window_samples} samples)")
        print(f"  Confidence threshold: {confidence_threshold}")

    def _load_model(self, model_path: str):
        """Load the trained Keras model."""
        try:
            import tensorflow as tf
            # Suppress TF warnings
            tf.get_logger().setLevel('ERROR')

            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path)
                print(f"  Model loaded successfully")
            else:
                print(f"  WARNING: Model not found at {model_path}")
                print(f"  Run train_beep_model.py first!")
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            self.model = None

    def extract_mfcc(self, samples: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio samples."""
        import librosa

        # Convert to float
        y = samples.astype(np.float32) / 32768.0

        # Extract MFCC
        hop_length = int(self.sample_rate * self.hop_duration_ms / 1000)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=hop_length
        )

        # Transpose to (n_frames, n_mfcc)
        return mfcc.T

    def detect(self, samples: np.ndarray) -> dict:
        """
        Run detection on audio samples.

        Returns dict with detection result and confidence.
        """
        if self.model is None:
            return {"detected": False, "confidence": 0.0, "error": "Model not loaded"}

        # Add samples to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])

        # Check if we have enough for a window
        if len(self.audio_buffer) < self.window_samples:
            return {"detected": False, "confidence": 0.0, "buffering": True}

        # Extract window (most recent samples)
        window = self.audio_buffer[-self.window_samples:]

        # Extract MFCC features
        mfcc = self.extract_mfcc(window)

        # Expected shape from training
        expected_frames = 50  # From training script

        # Pad or truncate
        if len(mfcc) < expected_frames:
            mfcc = np.pad(mfcc, ((0, expected_frames - len(mfcc)), (0, 0)))
        elif len(mfcc) > expected_frames:
            mfcc = mfcc[:expected_frames]

        # Reshape for model: (batch, frames, features)
        mfcc_input = mfcc.reshape(1, expected_frames, self.n_mfcc)

        # Run inference
        prediction = self.model.predict(mfcc_input, verbose=0)[0][0]

        # Threshold
        is_beep = prediction > self.confidence_threshold

        # Debounce
        if is_beep:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        confirmed = self.consecutive_detections >= self.debounce_count

        # Trim buffer to prevent memory growth (keep last window + some overlap)
        max_buffer = self.window_samples * 2
        if len(self.audio_buffer) > max_buffer:
            self.audio_buffer = self.audio_buffer[-max_buffer:]

        return {
            "detected": confirmed,
            "confidence": float(prediction),
            "raw_detection": is_beep,
            "consecutive": self.consecutive_detections,
        }

    def analyze_file(self, audio_path: str) -> list[dict]:
        """
        Analyze an audio file and return detections with timestamps.
        """
        import librosa

        print(f"\nAnalyzing file: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        print(f"  Duration: {len(y) / sr:.2f}s")
        print(f"  Samples: {len(y)}")

        # Convert to int16
        samples = (y * 32768).astype(np.int16)

        results = []
        detections = []

        # Slide window across audio
        hop = self.window_samples // 2  # 50% overlap
        for i in range(0, len(samples) - self.window_samples, hop):
            window = samples[i:i + self.window_samples]

            # Reset buffer for clean detection
            self.audio_buffer = np.array([], dtype=np.int16)
            self.consecutive_detections = 0

            # Run detection
            result = self.detect(window)
            result["timestamp_ms"] = (i / sr) * 1000
            result["timestamp_s"] = i / sr
            results.append(result)

            if result["confidence"] > self.confidence_threshold:
                detections.append(result)

        # Print summary
        print(f"\n  Analysis complete:")
        print(f"  Windows analyzed: {len(results)}")
        print(f"  Detections (confidence > {self.confidence_threshold}): {len(detections)}")

        if detections:
            print(f"\n  Detection timestamps:")
            for d in detections:
                print(f"    {d['timestamp_s']:.2f}s: confidence={d['confidence']:.3f}")

        return results


class AudioStreamServer:
    """UDP server that receives and analyzes audio from ESP32."""

    def __init__(
        self,
        port: int = 5000,
        sample_rate: int = 16000,
        target_freq: float = 2615.0,
        record_dir: str = "recordings",
        mqtt_host: Optional[str] = None,
        mqtt_port: int = 1883,
    ):
        self.port = port
        self.sample_rate = sample_rate
        self.target_freq = target_freq
        self.record_dir = record_dir
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port

        # Create analyzer
        self.analyzer = AudioAnalyzer(
            sample_rate=sample_rate,
            target_freq=target_freq,
        )

        # UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Recording state
        self.recording = False
        self.record_buffer: list[np.ndarray] = []
        self.continuous_record = False
        self.continuous_buffer = deque(maxlen=sample_rate * 60)  # 1 minute circular buffer

        # Statistics
        self.packets_received = 0
        self.bytes_received = 0
        self.last_sequence = -1
        self.packets_lost = 0
        self.start_time = None

        # MQTT client (optional)
        self.mqtt_client = None
        if mqtt_host:
            self._setup_mqtt()

        # Ensure recording directory exists
        os.makedirs(record_dir, exist_ok=True)

    def _setup_mqtt(self):
        """Initialize MQTT client for Home Assistant integration."""
        try:
            import paho.mqtt.client as mqtt

            self.mqtt_client = mqtt.Client()
            self.mqtt_client.connect(self.mqtt_host, self.mqtt_port)
            self.mqtt_client.loop_start()
            print(f"[MQTT] Connected to {self.mqtt_host}:{self.mqtt_port}")
        except Exception as e:
            print(f"[MQTT] Failed to connect: {e}")
            self.mqtt_client = None

    def _publish_detection(self, energy: float, rms: float):
        """Publish detection event to MQTT."""
        if self.mqtt_client:
            import json

            payload = json.dumps(
                {
                    "energy": round(energy, 6),
                    "rms": round(rms, 6),
                    "timestamp": datetime.now().isoformat(),
                    "target_freq": self.target_freq,
                }
            )
            self.mqtt_client.publish("esphome/audio/detection", payload)

    def start(self):
        """Start the UDP server and begin receiving audio."""
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(1.0)  # Allow periodic checks
        self.start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"Audio Streaming Server Started")
        print(f"{'=' * 60}")
        print(f"  Listening on port: {self.port}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Target frequency: {self.target_freq} Hz")
        print(f"  Recording directory: {self.record_dir}")
        print(f"{'=' * 60}")
        print("\nCommands:")
        print("  Press Ctrl+C to stop")
        print("  Audio analysis will be printed in real-time")
        print(f"{'=' * 60}\n")

        try:
            self._receive_loop()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self._cleanup()

    def _receive_loop(self):
        """Main receive loop."""
        last_stats_time = time.time()
        stats_interval = 5.0  # Print stats every 5 seconds

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                self._process_packet(data, addr)

            except socket.timeout:
                pass  # Normal timeout, continue loop

            # Periodic stats
            now = time.time()
            if now - last_stats_time >= stats_interval:
                self._print_stats()
                last_stats_time = now

    def _process_packet(self, data: bytes, addr: tuple):
        """Process a received UDP packet."""
        if len(data) < 6:  # Minimum: 4 byte seq + 2 byte sample
            return

        # Extract sequence number (little-endian uint32)
        sequence = struct.unpack("<I", data[:4])[0]

        # Check for packet loss
        if self.last_sequence >= 0:
            expected = (self.last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                lost = (sequence - expected) & 0xFFFFFFFF
                if lost < 1000:  # Reasonable loss count
                    self.packets_lost += lost
        self.last_sequence = sequence

        # Extract audio samples (16-bit signed, little-endian)
        audio_data = data[4:]
        samples = np.frombuffer(audio_data, dtype=np.int16)

        self.packets_received += 1
        self.bytes_received += len(data)

        # Add to continuous buffer
        self.continuous_buffer.extend(samples)

        # Analyze audio
        analysis = self.analyzer.analyze(samples)

        # Print real-time analysis
        self._print_analysis(sequence, analysis)

        # Check for detection
        if analysis["detected"]:
            print(f"\n*** BEEP DETECTED! Energy={analysis['energy']:.4f}, RMS={analysis['rms']:.4f} ***\n")
            self._publish_detection(analysis["energy"], analysis["rms"])

            # Auto-save snippet on detection
            self._save_detection_snippet()

        # Add to recording buffer if recording
        if self.recording:
            self.record_buffer.append(samples)

    def _print_analysis(self, seq: int, analysis: dict):
        """Print real-time analysis results."""
        # Use color codes for terminal
        rms = analysis["rms"]
        energy = analysis["energy"]

        # Color based on levels
        if energy > self.analyzer.energy_threshold:
            color = "\033[92m"  # Green - above threshold
        elif energy > self.analyzer.energy_threshold * 0.5:
            color = "\033[93m"  # Yellow - approaching threshold
        else:
            color = "\033[0m"  # Normal

        # Create visual bar for energy
        bar_len = min(50, int(energy * 500))
        bar = "#" * bar_len + "-" * (50 - bar_len)

        print(
            f"{color}seq={seq:8d} | RMS={rms:.4f} | Energy={energy:.6f} | [{bar}]\033[0m",
            end="\r",
        )

    def _print_stats(self):
        """Print periodic statistics."""
        elapsed = time.time() - self.start_time
        pps = self.packets_received / elapsed if elapsed > 0 else 0
        kbps = (self.bytes_received * 8 / 1000) / elapsed if elapsed > 0 else 0
        loss_pct = (self.packets_lost / (self.packets_received + self.packets_lost) * 100) if self.packets_received > 0 else 0

        print(f"\n[STATS] Packets: {self.packets_received}, Rate: {pps:.1f} pkt/s, {kbps:.1f} kbps, Loss: {loss_pct:.1f}%\n")

    def _save_detection_snippet(self):
        """Save a snippet of audio around the detection."""
        if len(self.continuous_buffer) < self.sample_rate:
            return

        # Get last 2 seconds of audio
        snippet_samples = self.sample_rate * 2
        samples = np.array(list(self.continuous_buffer)[-snippet_samples:], dtype=np.int16)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.record_dir, f"detection_{timestamp}.wav")

        self._save_wav(filename, samples)
        print(f"\n[SAVED] Detection snippet: {filename}\n")

    def _save_wav(self, filename: str, samples: np.ndarray):
        """Save samples to a WAV file."""
        with wave.open(filename, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(self.sample_rate)
            wav.writeframes(samples.tobytes())

    def start_recording(self, filename: Optional[str] = None):
        """Start recording audio to a file."""
        self.recording = True
        self.record_buffer = []
        self.record_filename = filename or os.path.join(
            self.record_dir,
            f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        )
        print(f"\n[RECORDING] Started: {self.record_filename}\n")

    def stop_recording(self) -> str:
        """Stop recording and save to file."""
        self.recording = False

        if not self.record_buffer:
            print("[RECORDING] No data captured")
            return ""

        # Concatenate all samples
        all_samples = np.concatenate(self.record_buffer)
        self._save_wav(self.record_filename, all_samples)

        duration = len(all_samples) / self.sample_rate
        print(f"\n[RECORDING] Saved: {self.record_filename} ({duration:.1f}s)\n")

        return self.record_filename

    def save_buffer(self, seconds: float = 10.0) -> str:
        """Save the last N seconds from the circular buffer."""
        samples_needed = int(self.sample_rate * seconds)
        available = len(self.continuous_buffer)

        if available == 0:
            print("[BUFFER] No data in buffer")
            return ""

        samples_to_save = min(samples_needed, available)
        samples = np.array(list(self.continuous_buffer)[-samples_to_save:], dtype=np.int16)

        filename = os.path.join(
            self.record_dir,
            f"buffer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav",
        )
        self._save_wav(filename, samples)

        duration = len(samples) / self.sample_rate
        print(f"\n[BUFFER] Saved: {filename} ({duration:.1f}s)\n")

        return filename

    def _cleanup(self):
        """Clean up resources."""
        if self.recording:
            self.stop_recording()

        self.sock.close()

        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()

        print(f"\nFinal stats:")
        print(f"  Packets received: {self.packets_received}")
        print(f"  Packets lost: {self.packets_lost}")
        print(f"  Bytes received: {self.bytes_received}")


def test_on_file(audio_path: str, model_path: str, confidence_threshold: float = 0.5):
    """Test neural network detection on an audio file."""
    print("\n" + "=" * 60)
    print("Neural Network Beep Detection Test")
    print("=" * 60)

    detector = NeuralBeepDetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
    )

    if detector.model is None:
        print("\nERROR: Could not load model. Run train_beep_model.py first!")
        return 1

    results = detector.analyze_file(audio_path)

    # Count detections
    high_confidence = [r for r in results if r["confidence"] > 0.8]
    medium_confidence = [r for r in results if 0.5 < r["confidence"] <= 0.8]
    low_confidence = [r for r in results if 0.3 < r["confidence"] <= 0.5]

    print(f"\n" + "=" * 60)
    print("Detection Summary")
    print("=" * 60)
    print(f"  High confidence (>0.8): {len(high_confidence)}")
    print(f"  Medium confidence (0.5-0.8): {len(medium_confidence)}")
    print(f"  Low confidence (0.3-0.5): {len(low_confidence)}")

    if high_confidence:
        print(f"\n  High confidence detections:")
        for r in high_confidence:
            print(f"    {r['timestamp_s']:.2f}s - confidence: {r['confidence']:.3f}")

    print("\n" + "=" * 60)
    if high_confidence:
        print("SUCCESS: Beeps detected in audio file!")
    else:
        print("WARNING: No high-confidence beeps detected.")
        print("Try adjusting --confidence-threshold or retrain the model.")
    print("=" * 60)

    return 0 if high_confidence else 1


def main():
    parser = argparse.ArgumentParser(
        description="Audio Streaming Server for ESPHome Beep Detector"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="UDP port to listen on (default: 5000)"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--target-freq",
        type=float,
        default=2615.0,
        help="Target frequency for beep detection in Hz (default: 2615)",
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="recordings",
        help="Directory to save recordings (default: recordings)",
    )
    parser.add_argument(
        "--mqtt-host",
        type=str,
        default=None,
        help="MQTT broker host for Home Assistant integration",
    )
    parser.add_argument(
        "--mqtt-port", type=int, default=1883, help="MQTT broker port (default: 1883)"
    )
    parser.add_argument(
        "--use-nn",
        action="store_true",
        help="Use neural network for beep detection instead of Goertzel",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/beep_detector.keras",
        help="Path to trained Keras model (default: models/beep_detector.keras)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for NN detection (default: 0.5)",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="Test detection on an audio file instead of live UDP stream",
    )

    args = parser.parse_args()

    # Test on file mode
    if args.test_file:
        return test_on_file(
            audio_path=args.test_file,
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
        )

    # Live UDP server mode
    server = AudioStreamServer(
        port=args.port,
        sample_rate=args.sample_rate,
        target_freq=args.target_freq,
        record_dir=args.record_dir,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
    )

    server.start()


if __name__ == "__main__":
    exit(main() or 0)
