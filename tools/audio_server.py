#!/usr/bin/env python3
"""
Audio Streaming Server for ESPHome Beep Detector Debugging

Receives UDP audio stream from ESP32 and provides:
- Real-time audio analysis (RMS, Goertzel energy, FFT)
- WAV file recording
- Detection event logging
- MQTT integration for Home Assistant

Usage:
    python audio_server.py --port 5000 --sample-rate 16000 --target-freq 2615

Requirements:
    pip install numpy scipy paho-mqtt
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

import numpy as np

from audio_analyzer import AudioAnalyzer


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

    args = parser.parse_args()

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
    main()
