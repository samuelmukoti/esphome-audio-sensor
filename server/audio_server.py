#!/usr/bin/env python3
"""
Audio Streaming Server for ESPHome Beep Detector

Receives UDP audio stream from ESP32, performs neural network-based
beep detection, and sends detection results back to ESP32.

Includes a web dashboard for:
- Real-time detection monitoring
- Labeling detections as True/False positives (reinforcement learning)
- Training mode to capture labeled samples
- Export labeled data for model fine-tuning

Architecture:
    ESP32 --UDP audio--> Server (NN inference) --UDP detection--> ESP32 --ESPHome API--> Home Assistant

Usage:
    # Start server with web dashboard
    python audio_server.py --port 5050 --web-port 8080

    # Open dashboard at http://localhost:8080

Requirements:
    pip install numpy scipy tensorflow librosa flask
"""

import argparse
import socket
import struct
import time
import wave
import os
import json
import threading
import uuid
from datetime import datetime
from collections import deque
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import numpy as np

# Flask for web dashboard
try:
    from flask import Flask, render_template_string, jsonify, request, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard disabled. Install with: pip install flask")


# ============================================
# Detection Event Storage for Labeling
# ============================================

class DetectionEvent:
    """Stores a detection event with audio for labeling."""

    def __init__(self, event_id: str, timestamp: datetime, confidence: float,
                 audio_samples: np.ndarray, sample_rate: int):
        self.id = event_id
        self.timestamp = timestamp
        self.confidence = confidence
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.label: Optional[bool] = None  # None=unlabeled, True=beep, False=not beep
        self.labeled_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "duration_ms": len(self.audio_samples) / self.sample_rate * 1000,
            "label": self.label,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
        }

    def save_audio(self, directory: str) -> str:
        """Save audio to WAV file and return path."""
        os.makedirs(directory, exist_ok=True)
        filename = f"{self.id}.wav"
        filepath = os.path.join(directory, filename)

        with wave.open(filepath, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wav.writeframes(self.audio_samples.tobytes())

        return filepath


class LabelingStore:
    """Manages detection events and labels for reinforcement learning."""

    def __init__(self, data_dir: str = "labeled_data"):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        self.events: Dict[str, DetectionEvent] = {}
        self.max_events = 500  # Keep more events in memory for training

        os.makedirs(self.audio_dir, exist_ok=True)
        self._load_labels()

    def _labels_file(self) -> str:
        return os.path.join(self.data_dir, "labels.json")

    def _load_labels(self):
        """Load existing labels from disk and restore events."""
        labels_file = self._labels_file()
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                data = json.load(f)
                labeled_data = data.get('labeled', [])

                # Restore events from saved labels
                for item in labeled_data:
                    event_id = item['id']
                    audio_path = os.path.join(self.audio_dir, f"{event_id}.wav")

                    # Only restore if audio file still exists
                    if os.path.exists(audio_path):
                        # Create event without audio samples (we'll load from disk when needed)
                        event = DetectionEvent(
                            event_id=event_id,
                            timestamp=datetime.fromisoformat(item['timestamp']),
                            confidence=item['confidence'],
                            audio_samples=np.array([], dtype=np.int16),  # Empty, loaded from file
                            sample_rate=16000,
                        )
                        event.label = item['label']
                        if item.get('labeled_at'):
                            event.labeled_at = datetime.fromisoformat(item['labeled_at'])

                        self.events[event_id] = event

                print(f"[LABELING] Restored {len(self.events)} labeled samples from disk")

    def _save_labels(self):
        """Save labels to disk."""
        labeled = [e.to_dict() for e in self.events.values() if e.label is not None]
        with open(self._labels_file(), 'w') as f:
            json.dump({"labeled": labeled}, f, indent=2)

    def add_event(self, confidence: float, audio_samples: np.ndarray,
                  sample_rate: int) -> DetectionEvent:
        """Add a new detection event."""
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        event = DetectionEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            confidence=confidence,
            audio_samples=audio_samples.copy(),
            sample_rate=sample_rate,
        )

        # Save audio immediately
        event.save_audio(self.audio_dir)

        # Add to memory store
        self.events[event_id] = event

        # Prune old UNLABELED events only (never delete labeled samples!)
        unlabeled = [e for e in self.events.values() if e.label is None]
        if len(unlabeled) > self.max_events:
            unlabeled.sort(key=lambda e: e.timestamp)
            # Keep the newest max_events/2, delete oldest
            to_delete = unlabeled[:len(unlabeled) - self.max_events // 2]
            for e in to_delete:
                if e.id in self.events:
                    # Also delete audio file for unlabeled samples
                    audio_path = os.path.join(self.audio_dir, f"{e.id}.wav")
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    del self.events[e.id]

        return event

    def label_event(self, event_id: str, is_beep: bool) -> bool:
        """Label an event as true/false positive."""
        if event_id not in self.events:
            return False

        event = self.events[event_id]
        event.label = is_beep
        event.labeled_at = datetime.now()

        self._save_labels()
        return True

    def get_recent_events(self, limit: int = 20, unlabeled_only: bool = False) -> List[dict]:
        """Get recent events for display."""
        events = self.events.values()
        if unlabeled_only:
            events = [e for e in events if e.label is None]
        events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return [e.to_dict() for e in events[:limit]]

    def get_stats(self) -> dict:
        """Get labeling statistics."""
        labeled = [e for e in self.events.values() if e.label is not None]
        # True positives: detector fired AND user confirmed it's a beep
        true_positives = sum(1 for e in labeled if e.label is True and e.confidence > 0)
        # False positives: detector fired BUT user said it's not a beep
        false_positives = sum(1 for e in labeled if e.label is False)
        # False negatives: confidence=0 means manual capture (detector missed it)
        false_negatives = sum(1 for e in labeled if e.label is True and e.confidence == 0)

        return {
            "total_events": len(self.events),
            "labeled": len(labeled),
            "unlabeled": len(self.events) - len(labeled),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": true_positives / len(labeled) if labeled else 0,
        }

    def export_training_data(self) -> str:
        """Export labeled data for retraining."""
        export_dir = os.path.join(self.data_dir, "export")
        beep_dir = os.path.join(export_dir, "beep")
        not_beep_dir = os.path.join(export_dir, "not_beep")

        os.makedirs(beep_dir, exist_ok=True)
        os.makedirs(not_beep_dir, exist_ok=True)

        exported = 0
        for event in self.events.values():
            if event.label is None:
                continue

            src_file = os.path.join(self.audio_dir, f"{event.id}.wav")
            if not os.path.exists(src_file):
                continue

            dest_dir = beep_dir if event.label else not_beep_dir
            dest_file = os.path.join(dest_dir, f"{event.id}.wav")

            # Copy file
            import shutil
            shutil.copy2(src_file, dest_file)
            exported += 1

        print(f"[EXPORT] Exported {exported} labeled samples to {export_dir}")
        return export_dir

    def retrain_model(self, base_model_path: str, output_model_path: str,
                      sample_rate: int = 16000, epochs: int = 20) -> dict:
        """Retrain the model with labeled samples for active learning."""
        import librosa
        import tensorflow as tf
        from tensorflow import keras
        from keras import layers

        print(f"\n{'='*60}")
        print("ACTIVE LEARNING: Retraining Model")
        print(f"{'='*60}")

        # Collect labeled samples
        labeled = [e for e in self.events.values() if e.label is not None]
        if len(labeled) < 2:
            return {"success": False, "error": "Need at least 2 labeled samples"}

        beeps = [e for e in labeled if e.label is True]
        not_beeps = [e for e in labeled if e.label is False]

        print(f"Labeled samples: {len(beeps)} beeps, {len(not_beeps)} not beeps")

        # Extract MFCC features from labeled audio
        X_new = []
        y_new = []
        n_mfcc = 20
        hop_length = 160  # 10ms at 16kHz
        expected_frames = 50  # 500ms window

        def extract_mfcc(audio, sr):
            """Extract MFCC features from audio."""
            center_samples = int(sr * 0.5)
            if len(audio) >= center_samples:
                start = (len(audio) - center_samples) // 2
                audio = audio[start:start + center_samples]

            mfcc = librosa.feature.mfcc(
                y=audio, sr=sr,
                n_mfcc=n_mfcc, n_fft=2048, hop_length=hop_length
            ).T

            if len(mfcc) < expected_frames:
                mfcc = np.pad(mfcc, ((0, expected_frames - len(mfcc)), (0, 0)))
            else:
                mfcc = mfcc[:expected_frames]
            return mfcc

        def augment_audio(audio, sr):
            """Generate augmented versions of audio for data augmentation."""
            augmented = []

            # Original
            augmented.append(audio)

            # Time shift (shift by up to 10% of duration)
            shift_max = int(len(audio) * 0.1)
            for shift in [-shift_max, shift_max]:
                shifted = np.roll(audio, shift)
                augmented.append(shifted)

            # Add noise (low level)
            noise = np.random.randn(len(audio)) * 0.005
            augmented.append(audio + noise)

            # Volume variation
            augmented.append(audio * 0.8)  # Quieter
            augmented.append(audio * 1.2)  # Louder

            return augmented

        # Separate positive and negative samples
        positive_audio = []
        negative_audio = []

        for event in labeled:
            audio_path = os.path.join(self.audio_dir, f"{event.id}.wav")
            if not os.path.exists(audio_path):
                print(f"  Warning: Audio not found for {event.id}")
                continue

            try:
                y_audio, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
                if event.label:
                    positive_audio.append(y_audio)
                else:
                    negative_audio.append(y_audio)
            except Exception as e:
                print(f"  Error loading {event.id}: {e}")

        print(f"Loaded audio: {len(positive_audio)} positive, {len(negative_audio)} negative")

        # Process positive samples with LIGHT augmentation (2x: original + noise)
        print("Processing positive samples (light augmentation)...")
        for audio in positive_audio:
            try:
                # Original
                mfcc = extract_mfcc(audio, sample_rate)
                X_new.append(mfcc)
                y_new.append(1)

                # One augmented version (noise)
                noise = np.random.randn(len(audio)) * 0.003
                mfcc_aug = extract_mfcc(audio + noise, sample_rate)
                X_new.append(mfcc_aug)
                y_new.append(1)
            except Exception as e:
                print(f"  Error processing positive: {e}")

        n_pos_samples = sum(1 for y in y_new if y == 1)
        print(f"Positive samples: {n_pos_samples}")

        # Process negative samples with SAME augmentation for balance
        print("Processing negative samples (light augmentation)...")
        for audio in negative_audio:
            try:
                # Original
                mfcc = extract_mfcc(audio, sample_rate)
                X_new.append(mfcc)
                y_new.append(0)

                # One augmented version (noise)
                noise = np.random.randn(len(audio)) * 0.003
                mfcc_aug = extract_mfcc(audio + noise, sample_rate)
                X_new.append(mfcc_aug)
                y_new.append(0)
            except Exception as e:
                print(f"  Error processing negative: {e}")

        n_neg_samples = sum(1 for y in y_new if y == 0)
        print(f"Negative samples: {n_neg_samples}")
        print(f"Balance ratio: 1:{n_neg_samples/max(n_pos_samples,1):.1f}")

        if len(X_new) < 2:
            return {"success": False, "error": "Could not extract features from samples"}

        X_new = np.array(X_new)
        y_new = np.array(y_new)

        print(f"New training data: {X_new.shape}")

        # Load base model or create new one
        if os.path.exists(base_model_path):
            print(f"Loading base model: {base_model_path}")
            model = keras.models.load_model(base_model_path)
        else:
            print("Creating new model (no base model found)")
            input_shape = (expected_frames, n_mfcc)
            model = keras.Sequential([
                layers.Input(shape=input_shape),
                layers.Conv1D(8, kernel_size=3, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.MaxPooling1D(pool_size=2),
                layers.Conv1D(8, kernel_size=3, padding='same'),
                layers.BatchNormalization(),
                layers.ReLU(),
                layers.GlobalAveragePooling1D(),
                layers.Dense(8),
                layers.ReLU(),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])

        # Compile with lower learning rate for fine-tuning
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Calculate class weights
        n_pos = np.sum(y_new == 1)
        n_neg = np.sum(y_new == 0)
        total = n_pos + n_neg
        class_weight = {
            0: total / (2 * n_neg) if n_neg > 0 else 1.0,
            1: total / (2 * n_pos) if n_pos > 0 else 1.0
        }

        print(f"Class weights: {class_weight}")
        print(f"Training for {epochs} epochs...")

        # Train (fine-tune)
        history = model.fit(
            X_new, y_new,
            epochs=epochs,
            batch_size=min(8, len(X_new)),
            class_weight=class_weight,
            validation_split=0.2 if len(X_new) >= 10 else 0.0,
            verbose=1
        )

        # Save model
        os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        model.save(output_model_path)
        print(f"Model saved to: {output_model_path}")

        # Get final metrics
        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]

        print(f"\n{'='*60}")
        print(f"Training complete! Loss: {final_loss:.4f}, Accuracy: {final_acc:.4f}")
        print(f"{'='*60}\n")

        return {
            "success": True,
            "model_path": output_model_path,
            "samples_used": len(X_new),
            "beeps": int(n_pos),
            "not_beeps": int(n_neg),
            "final_loss": float(final_loss),
            "final_accuracy": float(final_acc),
        }


# ============================================
# Neural Network Detector
# ============================================

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
        debounce_count: int = 2,
    ):
        self.sample_rate = sample_rate
        self.window_duration_ms = window_duration_ms
        self.hop_duration_ms = hop_duration_ms
        self.n_mfcc = n_mfcc
        self.confidence_threshold = confidence_threshold

        # Calculate window size in samples
        self.window_samples = int(sample_rate * window_duration_ms / 1000)
        self.hop_samples = int(sample_rate * hop_duration_ms / 1000)

        # Calculate expected MFCC frames based on window duration
        self.expected_frames = window_duration_ms // hop_duration_ms

        # Buffer for accumulating audio
        self.audio_buffer = np.array([], dtype=np.int16)

        # Load model
        self.model = None
        self._load_model(model_path)

        # Detection state for debouncing
        self.consecutive_detections = 0
        self.debounce_count = debounce_count
        self.detection_count = 0

        print(f"NeuralBeepDetector initialized:")
        print(f"  Model: {model_path}")
        print(f"  Window: {window_duration_ms}ms ({self.window_samples} samples)")
        print(f"  Expected MFCC frames: {self.expected_frames}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Debounce count: {debounce_count}")

    def _load_model(self, model_path: str):
        """Load the trained Keras model."""
        self.model_path = model_path
        try:
            import tensorflow as tf
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

    def reload_model(self, model_path: str = None):
        """Hot-reload the model (for active learning updates)."""
        if model_path is None:
            model_path = self.model_path
        print(f"\n[MODEL] Hot-reloading model from {model_path}...")
        self._load_model(model_path)
        # Reset detection state
        self.consecutive_detections = 0
        self.audio_buffer = np.array([], dtype=np.int16)
        print(f"[MODEL] Model reloaded successfully!\n")

    def extract_mfcc(self, samples: np.ndarray) -> np.ndarray:
        """Extract MFCC features from audio samples."""
        import librosa

        y = samples.astype(np.float32) / 32768.0
        hop_length = int(self.sample_rate * self.hop_duration_ms / 1000)
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=hop_length
        )
        return mfcc.T

    def detect(self, samples: np.ndarray) -> dict:
        """Run detection on audio samples."""
        if self.model is None:
            return {"detected": False, "confidence": 0.0, "error": "Model not loaded"}

        self.audio_buffer = np.concatenate([self.audio_buffer, samples])

        if len(self.audio_buffer) < self.window_samples:
            return {"detected": False, "confidence": 0.0, "buffering": True}

        window = self.audio_buffer[-self.window_samples:]
        mfcc = self.extract_mfcc(window)
        expected_frames = 50

        if len(mfcc) < expected_frames:
            mfcc = np.pad(mfcc, ((0, expected_frames - len(mfcc)), (0, 0)))
        elif len(mfcc) > expected_frames:
            mfcc = mfcc[:expected_frames]

        mfcc_input = mfcc.reshape(1, expected_frames, self.n_mfcc)
        prediction = self.model.predict(mfcc_input, verbose=0)[0][0]

        is_beep = prediction > self.confidence_threshold

        if is_beep:
            self.consecutive_detections += 1
        else:
            self.consecutive_detections = 0

        confirmed = self.consecutive_detections >= self.debounce_count

        max_buffer = self.window_samples * 2
        if len(self.audio_buffer) > max_buffer:
            self.audio_buffer = self.audio_buffer[-max_buffer:]

        return {
            "detected": confirmed,
            "confidence": float(prediction),
            "raw_detection": is_beep,
            "consecutive": self.consecutive_detections,
        }

    def analyze_file(self, audio_path: str) -> list:
        """Analyze an audio file and return detections with timestamps."""
        import librosa

        print(f"\nAnalyzing file: {audio_path}")
        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        print(f"  Duration: {len(y) / sr:.2f}s")

        samples = (y * 32768).astype(np.int16)
        results = []

        hop = self.window_samples // 2
        for i in range(0, len(samples) - self.window_samples, hop):
            window = samples[i:i + self.window_samples]
            self.audio_buffer = np.array([], dtype=np.int16)
            self.consecutive_detections = 0

            result = self.detect(window)
            result["timestamp_ms"] = (i / sr) * 1000
            result["timestamp_s"] = i / sr
            results.append(result)

        return results


# ============================================
# Web Dashboard
# ============================================

DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>Beep Detector - Training Dashboard</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #00d4ff; margin-bottom: 20px; }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .stat-value { font-size: 2em; color: #00d4ff; font-weight: bold; }
        .stat-label { color: #888; font-size: 0.9em; margin-top: 5px; }

        .mode-toggle {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .mode-toggle button {
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-training { background: #e94560; color: white; }
        .btn-training:hover { background: #ff6b6b; }
        .btn-training.active { background: #00d4ff; box-shadow: 0 0 20px #00d4ff; }
        .btn-mark-beep {
            background: linear-gradient(135deg, #ff6b6b, #ffa500);
            color: white;
            font-size: 1.2em;
            animation: glow 2s ease-in-out infinite;
        }
        .btn-mark-beep:hover {
            transform: scale(1.05);
            box-shadow: 0 0 30px rgba(255, 107, 107, 0.6);
        }
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 5px rgba(255, 107, 107, 0.5); }
            50% { box-shadow: 0 0 20px rgba(255, 107, 107, 0.8); }
        }
        .btn-retrain {
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            color: #000;
            font-weight: bold;
        }
        .btn-retrain:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        .btn-retrain:disabled {
            background: #444;
            color: #888;
            cursor: not-allowed;
            animation: none;
        }

        .status-indicator {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #333;
            animation: pulse 2s infinite;
        }
        .status-indicator.connected { background: #00ff88; }
        .status-indicator.detecting { background: #ff6b6b; animation: flash 0.3s infinite; }

        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes flash { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

        .events-section { margin-top: 30px; }
        .events-section h2 { color: #00d4ff; margin-bottom: 15px; }

        .event-card {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .event-card.labeled-true { border-left: 4px solid #00ff88; }
        .event-card.labeled-false { border-left: 4px solid #ff6b6b; }
        .event-card.unlabeled { border-left: 4px solid #666; }

        .event-info { flex: 1; }
        .event-time { color: #888; font-size: 0.9em; }
        .event-confidence {
            font-size: 1.5em;
            font-weight: bold;
            color: #00d4ff;
        }
        .confidence-bar {
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.3s;
        }

        .event-actions { display: flex; gap: 10px; }
        .event-actions button {
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }
        .btn-correct { background: #00ff88; color: #000; }
        .btn-correct:hover { transform: scale(1.05); }
        .btn-incorrect { background: #ff6b6b; color: #000; }
        .btn-incorrect:hover { transform: scale(1.05); }
        .btn-play { background: #0f3460; color: white; }
        .btn-play:hover { background: #1a5f7a; }

        .label-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .label-badge.true { background: #00ff88; color: #000; }
        .label-badge.false { background: #ff6b6b; color: #000; }
        .label-badge.manual { background: #ffa500; color: #000; }
        .label-badge.pending { background: #666; color: #fff; }
        .pending-badge { min-width: 120px; text-align: center; }

        .event-card.manual-capture {
            border-left: 4px solid #ffa500;
            background: linear-gradient(90deg, rgba(255, 165, 0, 0.1), #16213e);
        }
        .manual-indicator {
            color: #ffa500;
            font-size: 0.8em;
            margin-top: 5px;
        }

        .live-confidence {
            background: #16213e;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .live-value {
            font-size: 4em;
            font-weight: bold;
            color: #00d4ff;
        }
        .live-bar {
            height: 30px;
            background: #333;
            border-radius: 15px;
            margin-top: 20px;
            overflow: hidden;
        }
        .live-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d4ff, #00ff88);
            transition: width 0.1s;
        }

        .audio-player { margin-top: 10px; }
        audio { width: 100%; height: 40px; }

        .spectrum-section {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .spectrum-section h2 { color: #00d4ff; margin-bottom: 15px; }
        .spectrum-section h3 { color: #888; font-size: 0.9em; margin: 15px 0 10px; }
        .spectrum-container {
            background: #0a0a1a;
            border-radius: 8px;
            padding: 10px;
            position: relative;
        }
        #spectrum-canvas {
            width: 100%;
            height: 200px;
            border-radius: 4px;
        }
        .mfcc-container {
            background: #0a0a1a;
            border-radius: 8px;
            padding: 10px;
            margin-top: 15px;
        }
        #mfcc-canvas {
            width: 100%;
            height: 100px;
        }
        .audio-stats {
            display: flex;
            gap: 30px;
            margin-top: 15px;
            color: #888;
            font-size: 0.9em;
        }
        .audio-stats span {
            padding: 5px 15px;
            background: #0a0a1a;
            border-radius: 5px;
        }

        /* Workflow Guide */
        .workflow-guide {
            background: linear-gradient(135deg, #1a1a3a, #16213e);
            border: 1px solid #00d4ff33;
            border-radius: 10px;
            padding: 15px 20px;
            margin-bottom: 20px;
        }
        .workflow-guide h3 {
            color: #00d4ff;
            margin: 0 0 15px 0;
            font-size: 1em;
        }
        .workflow-scenarios {
            display: flex;
            gap: 20px;
        }
        .scenario {
            flex: 1;
            display: flex;
            gap: 12px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        .scenario-icon {
            font-size: 1.5em;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(0, 212, 255, 0.15);
            border-radius: 8px;
        }
        .scenario-content strong {
            color: #fff;
            display: block;
            margin-bottom: 5px;
        }
        .scenario-content p {
            color: #888;
            font-size: 0.85em;
            margin: 0;
        }
        .workflow-note {
            color: #666;
            font-size: 0.85em;
            margin: 12px 0 0 0;
            text-align: center;
            font-style: italic;
        }

        /* Section Help Text */
        .section-help {
            color: #888;
            font-size: 0.9em;
            margin: 5px 0 15px 0;
            padding: 10px 15px;
            background: rgba(0, 212, 255, 0.1);
            border-left: 3px solid #00d4ff;
            border-radius: 0 5px 5px 0;
        }
        .empty-state {
            text-align: center;
            padding: 30px;
            color: #666;
            font-style: italic;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 8px;
            border: 1px dashed #333;
        }

        /* Training Dataset Section */
        .dataset-section {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .dataset-section h2 { color: #00d4ff; margin-bottom: 15px; }
        .dataset-summary {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .dataset-stat {
            flex: 1;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .dataset-stat.positive { background: rgba(0, 255, 136, 0.2); border: 2px solid #00ff88; }
        .dataset-stat.negative { background: rgba(255, 107, 107, 0.2); border: 2px solid #ff6b6b; }
        .dataset-stat.total { background: rgba(0, 212, 255, 0.2); border: 2px solid #00d4ff; }
        .dataset-stat .count { display: block; font-size: 2em; font-weight: bold; }
        .dataset-stat.positive .count { color: #00ff88; }
        .dataset-stat.negative .count { color: #ff6b6b; }
        .dataset-stat.total .count { color: #00d4ff; }
        .dataset-stat .label { color: #888; font-size: 0.9em; }

        .dataset-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .tab-btn {
            padding: 8px 20px;
            border: none;
            border-radius: 5px;
            background: #0a0a1a;
            color: #888;
            cursor: pointer;
            transition: all 0.2s;
        }
        .tab-btn:hover { background: #1a1a3a; color: #fff; }
        .tab-btn.active { background: #00d4ff; color: #000; font-weight: bold; }

        .dataset-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .sample-card {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 12px 15px;
            background: #0a0a1a;
            border-radius: 8px;
            margin-bottom: 8px;
            border-left: 4px solid #666;
        }
        .sample-card.positive { border-left-color: #00ff88; }
        .sample-card.negative { border-left-color: #ff6b6b; }
        .sample-type {
            width: 80px;
            text-align: center;
            padding: 5px 10px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 0.85em;
        }
        .sample-type.positive { background: #00ff88; color: #000; }
        .sample-type.negative { background: #ff6b6b; color: #000; }
        .sample-info { flex: 1; }
        .sample-id { font-family: monospace; color: #666; font-size: 0.8em; }
        .sample-meta { color: #888; font-size: 0.85em; margin-top: 3px; }
        .sample-actions { display: flex; gap: 8px; }
        .btn-play-sm {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            background: #0f3460;
            color: white;
            cursor: pointer;
        }
        .btn-play-sm:hover { background: #1a5f7a; }
        .btn-delete {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            background: #8b0000;
            color: white;
            cursor: pointer;
        }
        .btn-delete:hover { background: #a00; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔊 Beep Detector - Training Dashboard</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value" id="total-detections">0</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="labeled-count">0</div>
                <div class="stat-label">Labeled</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="true-positives">0</div>
                <div class="stat-label">True Positives</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="false-positives">0</div>
                <div class="stat-label">False Positives</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="precision">-</div>
                <div class="stat-label">Precision</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="false-negatives">0</div>
                <div class="stat-label">False Negatives</div>
            </div>
        </div>

        <div class="mode-toggle">
            <div class="status-indicator" id="status-indicator"></div>
            <span id="status-text">Waiting for connection...</span>
            <button class="btn-training" id="training-btn" onclick="toggleTraining()">
                🎯 Training Mode: OFF
            </button>
            <button class="btn-mark-beep" id="mark-beep-btn" onclick="markBeepNow()">
                🔔 Mark Beep NOW
            </button>
            <button class="btn-retrain" id="retrain-btn" onclick="retrainModel()">
                🧠 Retrain Model
            </button>
        </div>

        <div class="live-confidence">
            <div>Live Confidence</div>
            <div class="live-value" id="live-confidence">0.00</div>
            <div class="live-bar">
                <div class="live-fill" id="live-bar" style="width: 0%"></div>
            </div>
        </div>

        <div class="workflow-guide">
            <h3>📋 Labeling Workflow</h3>
            <div class="workflow-scenarios">
                <div class="scenario">
                    <div class="scenario-icon">🔔</div>
                    <div class="scenario-content">
                        <strong>Beep happened but model MISSED it?</strong>
                        <p>Click "Mark Beep NOW" while the beep is happening to capture it as a positive sample.</p>
                    </div>
                </div>
                <div class="scenario">
                    <div class="scenario-icon">⏳</div>
                    <div class="scenario-content">
                        <strong>Model detected something?</strong>
                        <p>Review samples in "Pending Review" below. Listen and click "Beep" or "Not Beep" to label them.</p>
                    </div>
                </div>
            </div>
            <p class="workflow-note">💡 No "Not Beep NOW" needed: if the model doesn't detect anything, it's already correct!</p>
        </div>

        <div class="spectrum-section">
            <h2>🎵 Audio Spectrum Analyzer</h2>
            <div class="spectrum-container">
                <canvas id="spectrum-canvas" width="800" height="200"></canvas>
            </div>
            <div class="mfcc-container">
                <h3>MFCC Features (Model Input)</h3>
                <canvas id="mfcc-canvas" width="400" height="100"></canvas>
            </div>
            <div class="audio-stats">
                <span id="rms-level">RMS: --</span>
                <span id="peak-freq">Peak: -- Hz</span>
            </div>
        </div>

        <div class="events-section">
            <h2>⏳ Pending Review</h2>
            <p class="section-help">These samples need your label. Click "Beep" or "Not Beep" to add them to the training dataset.</p>
            <div id="events-list"></div>
            <div id="no-pending" class="empty-state" style="display:none;">
                ✅ No samples pending review. Samples appear here when the model detects potential beeps.
            </div>
        </div>

        <div class="dataset-section">
            <h2>📚 Training Dataset (Used for Retraining)</h2>
            <div class="dataset-summary">
                <div class="dataset-stat positive">
                    <span class="count" id="dataset-positive">0</span>
                    <span class="label">✓ Beep Samples</span>
                </div>
                <div class="dataset-stat negative">
                    <span class="count" id="dataset-negative">0</span>
                    <span class="label">✗ Not Beep Samples</span>
                </div>
                <div class="dataset-stat total">
                    <span class="count" id="dataset-total">0</span>
                    <span class="label">Total Samples</span>
                </div>
            </div>
            <div class="dataset-tabs">
                <button class="tab-btn active" onclick="showDatasetTab('all')">All</button>
                <button class="tab-btn" onclick="showDatasetTab('positive')">Beep Only</button>
                <button class="tab-btn" onclick="showDatasetTab('negative')">Not Beep Only</button>
            </div>
            <div id="dataset-list" class="dataset-list"></div>
        </div>
    </div>

    <script>
        let trainingMode = false;

        function toggleTraining() {
            trainingMode = !trainingMode;
            const btn = document.getElementById('training-btn');
            btn.textContent = trainingMode ? '🎯 Training Mode: ON' : '🎯 Training Mode: OFF';
            btn.classList.toggle('active', trainingMode);

            fetch('/api/training-mode', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled: trainingMode})
            });
        }

        function labelEvent(eventId, isBeep) {
            fetch('/api/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({event_id: eventId, is_beep: isBeep})
            }).then(() => fetchEvents());
        }

        function playAudio(eventId) {
            const audio = document.getElementById('audio-' + eventId);
            if (audio) audio.play();
        }

        function retrainModel() {
            const btn = document.getElementById('retrain-btn');
            const originalText = btn.textContent;
            btn.textContent = '⏳ Training...';
            btn.disabled = true;

            fetch('/api/retrain', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        btn.textContent = '✓ Model Updated!';
                        btn.style.background = '#00ff88';
                        alert(`Model retrained successfully!\n\nSamples used: ${data.samples_used}\nBeeps: ${data.beeps}\nNot beeps: ${data.not_beeps}\nAccuracy: ${(data.final_accuracy * 100).toFixed(1)}%`);
                    } else {
                        btn.textContent = '❌ ' + (data.error || 'Failed');
                        alert('Training failed: ' + (data.error || 'Unknown error'));
                    }
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = '';
                        btn.disabled = false;
                    }, 3000);
                })
                .catch(err => {
                    btn.textContent = '❌ Error';
                    alert('Training error: ' + err);
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.disabled = false;
                    }, 3000);
                });
        }

        function markBeepNow() {
            const btn = document.getElementById('mark-beep-btn');
            btn.textContent = '⏳ Capturing...';
            btn.disabled = true;

            fetch('/api/mark-beep', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        btn.textContent = '✓ Captured!';
                        btn.style.background = '#00ff88';
                        setTimeout(() => {
                            btn.textContent = '🔔 Mark Beep NOW';
                            btn.style.background = '';
                            btn.disabled = false;
                        }, 1500);
                        fetchEvents();
                    } else {
                        btn.textContent = '❌ ' + data.error;
                        setTimeout(() => {
                            btn.textContent = '🔔 Mark Beep NOW';
                            btn.style.background = '';
                            btn.disabled = false;
                        }, 2000);
                    }
                })
                .catch(err => {
                    btn.textContent = '❌ Error';
                    setTimeout(() => {
                        btn.textContent = '🔔 Mark Beep NOW';
                        btn.disabled = false;
                    }, 2000);
                });
        }

        function fetchEvents() {
            fetch('/api/events')
                .then(r => r.json())
                .then(data => {
                    renderEvents(data.events);
                    updateStats(data.stats);
                });
        }

        function fetchStatus() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    const indicator = document.getElementById('status-indicator');
                    const text = document.getElementById('status-text');

                    if (data.connected) {
                        indicator.classList.add('connected');
                        text.textContent = 'Connected to ESP32 (' + data.esp32_ip + ')';

                        if (data.detecting) {
                            indicator.classList.add('detecting');
                        } else {
                            indicator.classList.remove('detecting');
                        }
                    } else {
                        indicator.classList.remove('connected', 'detecting');
                        text.textContent = 'Waiting for ESP32...';
                    }

                    document.getElementById('live-confidence').textContent = data.confidence.toFixed(3);
                    document.getElementById('live-bar').style.width = (data.confidence * 100) + '%';
                });
        }

        function updateStats(stats) {
            document.getElementById('total-detections').textContent = stats.total_events;
            document.getElementById('labeled-count').textContent = stats.labeled;
            document.getElementById('true-positives').textContent = stats.true_positives;
            document.getElementById('false-positives').textContent = stats.false_positives;
            document.getElementById('false-negatives').textContent = stats.false_negatives || 0;
            document.getElementById('precision').textContent =
                stats.labeled > 0 ? (stats.precision * 100).toFixed(1) + '%' : '-';
        }

        function renderEvents(events) {
            const container = document.getElementById('events-list');
            const emptyState = document.getElementById('no-pending');

            // Show/hide empty state
            if (events.length === 0) {
                container.style.display = 'none';
                emptyState.style.display = 'block';
                return;
            }
            container.style.display = 'block';
            emptyState.style.display = 'none';

            // These are all UNLABELED samples needing review
            container.innerHTML = events.map(e => {
                const confidenceDisplay = `Model confidence: ${(e.confidence * 100).toFixed(1)}%`;

                return `
                    <div class="event-card unlabeled">
                        <div class="event-info">
                            <div class="event-time">${new Date(e.timestamp).toLocaleString()}</div>
                            <div class="event-confidence">${confidenceDisplay}</div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${e.confidence * 100}%"></div>
                            </div>
                            <div class="audio-player">
                                <audio id="audio-${e.id}" src="/api/audio/${e.id}" preload="none"></audio>
                            </div>
                        </div>
                        <div class="pending-badge">
                            <span class="label-badge pending">⏳ Needs Label</span>
                        </div>
                        <div class="event-actions">
                            <button class="btn-play" onclick="playAudio('${e.id}')">▶ Play</button>
                            <button class="btn-correct" onclick="labelEvent('${e.id}', true)">✓ Beep</button>
                            <button class="btn-incorrect" onclick="labelEvent('${e.id}', false)">✗ Not Beep</button>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // Poll for updates
        setInterval(fetchStatus, 500);
        setInterval(fetchEvents, 2000);
        fetchEvents();

        // ========================================
        // Spectrum Analyzer Visualization
        // ========================================
        const spectrumCanvas = document.getElementById('spectrum-canvas');
        const spectrumCtx = spectrumCanvas.getContext('2d');
        const mfccCanvas = document.getElementById('mfcc-canvas');
        const mfccCtx = mfccCanvas.getContext('2d');

        // Smooth spectrum data
        let smoothedSpectrum = new Array(128).fill(0);
        const smoothingFactor = 0.3;

        function drawSpectrum(data) {
            const canvas = spectrumCanvas;
            const ctx = spectrumCtx;
            const width = canvas.width;
            const height = canvas.height;

            // Clear canvas
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, width, height);

            if (!data.spectrum || data.spectrum.length === 0) return;

            const spectrum = data.spectrum;
            const barWidth = width / spectrum.length;

            // Smooth the spectrum
            for (let i = 0; i < spectrum.length; i++) {
                smoothedSpectrum[i] = smoothedSpectrum[i] * (1 - smoothingFactor) + spectrum[i] * smoothingFactor;
            }

            // Draw frequency bands
            for (let i = 0; i < smoothedSpectrum.length; i++) {
                const value = smoothedSpectrum[i];
                const barHeight = value * height * 0.9;

                // Color gradient based on frequency and intensity
                const hue = 180 + (i / smoothedSpectrum.length) * 60; // Cyan to green
                const saturation = 80;
                const lightness = 30 + value * 40;

                ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
                ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);

                // Glow effect for high values
                if (value > 0.6) {
                    ctx.shadowColor = `hsl(${hue}, 100%, 50%)`;
                    ctx.shadowBlur = 10;
                    ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
                    ctx.shadowBlur = 0;
                }
            }

            // Draw frequency labels
            ctx.fillStyle = '#666';
            ctx.font = '10px monospace';
            const freqLabels = ['0', '1k', '2k', '4k', '6k', '8k'];
            freqLabels.forEach((label, i) => {
                const x = (i / (freqLabels.length - 1)) * width;
                ctx.fillText(label + 'Hz', x, height - 5);
            });

            // Draw detection threshold line
            const thresholdY = height * 0.3;
            ctx.strokeStyle = 'rgba(255, 107, 107, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(0, thresholdY);
            ctx.lineTo(width, thresholdY);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        function drawMFCC(mfccData) {
            const canvas = mfccCanvas;
            const ctx = mfccCtx;
            const width = canvas.width;
            const height = canvas.height;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, width, height);

            if (!mfccData || mfccData.length === 0) return;

            const barWidth = width / mfccData.length;
            const maxVal = Math.max(...mfccData.map(Math.abs)) || 1;

            for (let i = 0; i < mfccData.length; i++) {
                const value = mfccData[i] / maxVal;
                const barHeight = Math.abs(value) * height * 0.4;
                const y = value >= 0 ? height/2 - barHeight : height/2;

                const hue = value >= 0 ? 120 : 0; // Green for positive, red for negative
                ctx.fillStyle = `hsl(${hue}, 70%, ${40 + Math.abs(value) * 30}%)`;
                ctx.fillRect(i * barWidth, y, barWidth - 2, barHeight);
            }

            // Draw center line
            ctx.strokeStyle = '#333';
            ctx.beginPath();
            ctx.moveTo(0, height/2);
            ctx.lineTo(width, height/2);
            ctx.stroke();

            // Labels
            ctx.fillStyle = '#666';
            ctx.font = '10px monospace';
            ctx.fillText('MFCC Coefficients (C0-C19)', 5, 12);
        }

        function fetchSpectrum() {
            fetch('/api/spectrum')
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        drawSpectrum(data);
                        drawMFCC(data.mfcc);

                        // Update stats
                        document.getElementById('rms-level').textContent =
                            'RMS: ' + (data.rms * 100).toFixed(1) + '%';

                        // Find peak frequency
                        if (data.spectrum && data.frequencies) {
                            const maxIdx = data.spectrum.indexOf(Math.max(...data.spectrum));
                            const peakFreq = data.frequencies[maxIdx];
                            document.getElementById('peak-freq').textContent =
                                'Peak: ' + peakFreq.toFixed(0) + ' Hz';
                        }
                    }
                })
                .catch(() => {});
        }

        // Update spectrum at ~20 FPS
        setInterval(fetchSpectrum, 50);

        // ========================================
        // Training Dataset Management
        // ========================================
        let currentDatasetFilter = 'all';
        let datasetSamples = [];

        function showDatasetTab(filter) {
            currentDatasetFilter = filter;
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderDataset();
        }

        function fetchDataset() {
            fetch('/api/dataset')
                .then(r => r.json())
                .then(data => {
                    datasetSamples = data.samples;
                    document.getElementById('dataset-positive').textContent = data.stats.positive;
                    document.getElementById('dataset-negative').textContent = data.stats.negative;
                    document.getElementById('dataset-total').textContent = data.stats.total;
                    renderDataset();
                });
        }

        function renderDataset() {
            const container = document.getElementById('dataset-list');
            let filtered = datasetSamples;

            if (currentDatasetFilter === 'positive') {
                filtered = datasetSamples.filter(s => s.label === true);
            } else if (currentDatasetFilter === 'negative') {
                filtered = datasetSamples.filter(s => s.label === false);
            }

            if (filtered.length === 0) {
                container.innerHTML = '<div style="text-align:center;color:#666;padding:20px;">No samples in this category</div>';
                return;
            }

            container.innerHTML = filtered.map(s => {
                const typeClass = s.label ? 'positive' : 'negative';
                const typeText = s.label ? '✓ BEEP' : '✗ NOT BEEP';
                const source = s.confidence === 0 ? 'Manual capture' : `Detected (${(s.confidence * 100).toFixed(0)}%)`;
                const date = new Date(s.timestamp).toLocaleString();

                return `
                    <div class="sample-card ${typeClass}">
                        <div class="sample-type ${typeClass}">${typeText}</div>
                        <div class="sample-info">
                            <div class="sample-id">${s.id}</div>
                            <div class="sample-meta">${date} · ${source}</div>
                        </div>
                        <div class="sample-actions">
                            <button class="btn-play-sm" onclick="playSample('${s.id}')">▶ Play</button>
                            <button class="btn-delete" onclick="deleteSample('${s.id}')">🗑 Delete</button>
                        </div>
                        <audio id="sample-${s.id}" src="/api/audio/${s.id}" preload="none"></audio>
                    </div>
                `;
            }).join('');
        }

        function playSample(id) {
            const audio = document.getElementById('sample-' + id);
            if (audio) audio.play();
        }

        function deleteSample(id) {
            if (!confirm('Delete this sample from training dataset?')) return;

            fetch('/api/dataset/' + id, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        fetchDataset();
                        fetchEvents();
                    } else {
                        alert('Failed to delete: ' + data.error);
                    }
                });
        }

        // Fetch dataset on load and periodically
        fetchDataset();
        setInterval(fetchDataset, 5000);
    </script>
</body>
</html>
'''


# ============================================
# Audio Stream Server with Web Dashboard
# ============================================

class AudioStreamServer:
    """UDP server that receives audio from ESP32, runs NN inference, and sends results back."""

    def __init__(
        self,
        port: int = 5000,
        response_port: int = 5001,
        sample_rate: int = 16000,
        record_dir: str = "recordings",
        model_path: str = "models/beep_detector.keras",
        window_ms: int = 500,
        confidence_threshold: float = 0.5,
        web_port: int = 8080,
    ):
        self.port = port
        self.response_port = response_port
        self.sample_rate = sample_rate
        self.record_dir = record_dir
        self.web_port = web_port

        # Create neural network detector
        self.nn_detector = NeuralBeepDetector(
            model_path=model_path,
            sample_rate=sample_rate,
            window_duration_ms=window_ms,
            confidence_threshold=confidence_threshold,
        )

        # UDP sockets
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Track ESP32 client address for sending responses
        self.esp32_addr: Optional[Tuple[str, int]] = None

        # Recording state
        self.recording = False
        self.record_buffer: list = []
        self.continuous_buffer = deque(maxlen=sample_rate * 60)

        # Statistics
        self.packets_received = 0
        self.bytes_received = 0
        self.last_sequence = -1
        self.packets_lost = 0
        self.start_time = None
        self.nn_inferences = 0

        # Detection state tracking
        self.last_detection_state = False
        self.current_confidence = 0.0

        # Training/labeling mode
        self.training_mode = False
        self.labeling_store = LabelingStore(os.path.join(record_dir, "labeled_data"))

        # Ensure directories exist
        os.makedirs(record_dir, exist_ok=True)

        # Flask app for dashboard
        self.flask_app = None
        if FLASK_AVAILABLE:
            self._setup_flask()

    def _setup_flask(self):
        """Set up Flask web server for dashboard."""
        self.flask_app = Flask(__name__)
        self.flask_app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

        @self.flask_app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_HTML)

        @self.flask_app.route('/api/status')
        def api_status():
            return jsonify({
                "connected": self.esp32_addr is not None,
                "esp32_ip": self.esp32_addr[0] if self.esp32_addr else None,
                "detecting": self.last_detection_state,
                "confidence": self.current_confidence,
                "training_mode": self.training_mode,
            })

        @self.flask_app.route('/api/events')
        def api_events():
            # Return only UNLABELED events for the pending review list
            return jsonify({
                "events": self.labeling_store.get_recent_events(unlabeled_only=True),
                "stats": self.labeling_store.get_stats(),
            })

        @self.flask_app.route('/api/label', methods=['POST'])
        def api_label():
            data = request.json
            success = self.labeling_store.label_event(data['event_id'], data['is_beep'])
            return jsonify({"success": success})

        @self.flask_app.route('/api/training-mode', methods=['POST'])
        def api_training_mode():
            data = request.json
            self.training_mode = data.get('enabled', False)
            print(f"[TRAINING] Mode {'enabled' if self.training_mode else 'disabled'}")
            return jsonify({"enabled": self.training_mode})

        @self.flask_app.route('/api/export', methods=['POST'])
        def api_export():
            path = self.labeling_store.export_training_data()
            return jsonify({"path": path})

        @self.flask_app.route('/api/retrain', methods=['POST'])
        def api_retrain():
            """Retrain the model with labeled samples and hot-reload."""
            try:
                # Define model paths
                base_model = self.nn_detector.model_path
                output_model = os.path.join(
                    os.path.dirname(base_model) or "models",
                    "beep_detector_active.keras"
                )

                # Retrain
                result = self.labeling_store.retrain_model(
                    base_model_path=base_model,
                    output_model_path=output_model,
                    sample_rate=self.sample_rate,
                    epochs=20
                )

                if result["success"]:
                    # Hot-reload the new model
                    self.nn_detector.reload_model(output_model)
                    result["model_reloaded"] = True

                return jsonify(result)

            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    "success": False,
                    "error": str(e)
                })

        @self.flask_app.route('/api/mark-beep', methods=['POST'])
        def api_mark_beep():
            """Manually mark current audio as containing a beep (for false negatives)."""
            if len(self.continuous_buffer) < self.sample_rate * 2:
                return jsonify({
                    "success": False,
                    "error": "Not enough audio buffered yet"
                })

            # Capture 2 seconds of audio ending now
            audio_samples = np.array(
                list(self.continuous_buffer)[-self.sample_rate * 2:],
                dtype=np.int16
            )

            # Create event with confidence=0 to indicate manual capture
            # Pre-label as True (beep) since user is marking it as a beep
            event = self.labeling_store.add_event(
                confidence=0.0,  # 0 confidence = manual capture (model missed it)
                audio_samples=audio_samples,
                sample_rate=self.sample_rate,
            )

            # Auto-label as true positive since user clicked "Mark Beep Now"
            self.labeling_store.label_event(event.id, is_beep=True)

            print(f"\n*** MANUAL BEEP MARKED by user ***")
            print(f"    Saved: {event.id}")
            print(f"    This was a FALSE NEGATIVE - model missed this beep\n")

            return jsonify({
                "success": True,
                "event_id": event.id,
                "message": "Captured and labeled as beep (false negative)"
            })

        @self.flask_app.route('/api/audio/<event_id>')
        def api_audio(event_id):
            audio_path = os.path.join(self.labeling_store.audio_dir, f"{event_id}.wav")
            if os.path.exists(audio_path):
                return send_file(audio_path, mimetype='audio/wav')
            return "Not found", 404

        @self.flask_app.route('/api/spectrum')
        def api_spectrum():
            """Return current FFT spectrum and MFCC features for visualization."""
            from scipy.fft import rfft, rfftfreq

            if len(self.continuous_buffer) < 1024:
                return jsonify({
                    "success": False,
                    "error": "Not enough audio buffered"
                })

            # Get last 1024 samples (~64ms at 16kHz)
            samples = np.array(list(self.continuous_buffer)[-2048:], dtype=np.float32)
            samples = samples / 32768.0  # Normalize

            # Apply Hanning window
            window = np.hanning(len(samples))
            windowed = samples * window

            # Compute FFT
            fft_result = rfft(windowed)
            magnitudes = np.abs(fft_result)

            # Convert to dB scale
            magnitudes_db = 20 * np.log10(magnitudes + 1e-10)
            magnitudes_db = np.clip(magnitudes_db, -80, 0)  # Clip to -80dB floor
            magnitudes_normalized = (magnitudes_db + 80) / 80  # Normalize to 0-1

            # Get frequencies
            freqs = rfftfreq(len(samples), 1/self.sample_rate)

            # Downsample to ~128 bins for visualization
            num_bins = 128
            bin_size = len(magnitudes_normalized) // num_bins
            spectrum_bins = []
            freq_labels = []
            for i in range(num_bins):
                start = i * bin_size
                end = start + bin_size
                spectrum_bins.append(float(np.mean(magnitudes_normalized[start:end])))
                freq_labels.append(float(np.mean(freqs[start:end])))

            # Also return MFCC energies if available
            mfcc_energies = []
            try:
                import librosa
                mfcc = librosa.feature.mfcc(y=samples, sr=self.sample_rate, n_mfcc=20)
                mfcc_energies = mfcc[:, -1].tolist()  # Last frame
            except:
                pass

            return jsonify({
                "success": True,
                "spectrum": spectrum_bins,
                "frequencies": freq_labels,
                "mfcc": mfcc_energies,
                "confidence": self.current_confidence,
                "rms": float(np.sqrt(np.mean(samples ** 2))),
            })

        @self.flask_app.route('/api/dataset')
        def api_dataset():
            """Get all labeled samples for the training dataset view."""
            labeled = [e for e in self.labeling_store.events.values() if e.label is not None]
            labeled.sort(key=lambda e: e.timestamp, reverse=True)

            samples = [e.to_dict() for e in labeled]
            positive = sum(1 for e in labeled if e.label is True)
            negative = sum(1 for e in labeled if e.label is False)

            return jsonify({
                "samples": samples,
                "stats": {
                    "positive": positive,
                    "negative": negative,
                    "total": len(labeled)
                }
            })

        @self.flask_app.route('/api/dataset/<sample_id>', methods=['DELETE'])
        def api_delete_sample(sample_id):
            """Delete a sample from the training dataset."""
            if sample_id not in self.labeling_store.events:
                return jsonify({"success": False, "error": "Sample not found"})

            event = self.labeling_store.events[sample_id]

            # Delete audio file
            audio_path = os.path.join(self.labeling_store.audio_dir, f"{sample_id}.wav")
            if os.path.exists(audio_path):
                os.remove(audio_path)
                print(f"[DATASET] Deleted audio: {audio_path}")

            # Remove from memory
            del self.labeling_store.events[sample_id]

            # Update labels file
            self.labeling_store._save_labels()

            print(f"[DATASET] Removed sample {sample_id} from training dataset")

            return jsonify({"success": True})

    def _send_detection_to_esp32(self, detected: bool, confidence: float):
        """Send detection result back to ESP32 via UDP."""
        if self.esp32_addr is None:
            return

        packet = struct.pack("<Bf", 1 if detected else 0, confidence)

        try:
            target_addr = (self.esp32_addr[0], self.response_port)
            self.sock.sendto(packet, target_addr)
        except Exception as e:
            print(f"[ERROR] Failed to send detection to ESP32: {e}")

    def start(self):
        """Start the UDP server and web dashboard."""
        self.sock.bind(("0.0.0.0", self.port))
        self.sock.settimeout(1.0)
        self.start_time = time.time()

        print(f"\n{'=' * 60}")
        print(f"Audio Streaming Server - Neural Network Beep Detection")
        print(f"{'=' * 60}")
        print(f"  Audio receive port: {self.port}")
        print(f"  Detection response port: {self.response_port}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  NN window: {self.nn_detector.window_duration_ms}ms")
        print(f"  Confidence threshold: {self.nn_detector.confidence_threshold}")

        if FLASK_AVAILABLE:
            print(f"\n  📊 Web Dashboard: http://localhost:{self.web_port}")

        print(f"{'=' * 60}")
        print("\nArchitecture:")
        print("  ESP32 --UDP audio--> Server (NN) --UDP detection--> ESP32 --API--> HA")
        print(f"\nWaiting for audio stream from ESP32...")
        print(f"{'=' * 60}\n")

        # Start Flask in background thread
        if FLASK_AVAILABLE and self.flask_app:
            flask_thread = threading.Thread(
                target=lambda: self.flask_app.run(
                    host='0.0.0.0',
                    port=self.web_port,
                    debug=False,
                    use_reloader=False,
                    threaded=True
                ),
                daemon=True
            )
            flask_thread.start()
            print(f"[WEB] Dashboard started at http://localhost:{self.web_port}")

        try:
            self._receive_loop()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self._cleanup()

    def _receive_loop(self):
        """Main receive loop."""
        last_stats_time = time.time()
        stats_interval = 5.0

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                self._process_packet(data, addr)

            except socket.timeout:
                pass

            now = time.time()
            if now - last_stats_time >= stats_interval:
                self._print_stats()
                last_stats_time = now

    def _process_packet(self, data: bytes, addr: tuple):
        """Process a received UDP packet."""
        if len(data) < 6:
            return

        if self.esp32_addr is None or self.esp32_addr[0] != addr[0]:
            self.esp32_addr = addr
            print(f"[INFO] ESP32 connected from {addr[0]}:{addr[1]}")

        sequence = struct.unpack("<I", data[:4])[0]

        if self.last_sequence >= 0:
            expected = (self.last_sequence + 1) & 0xFFFFFFFF
            if sequence != expected:
                lost = (sequence - expected) & 0xFFFFFFFF
                if lost < 1000:
                    self.packets_lost += lost
        self.last_sequence = sequence

        audio_data = data[4:]
        samples = np.frombuffer(audio_data, dtype=np.int16)

        self.packets_received += 1
        self.bytes_received += len(data)
        self.continuous_buffer.extend(samples)

        nn_result = self.nn_detector.detect(samples)
        self.nn_inferences += 1

        if not nn_result.get("buffering", False):
            detected = nn_result["detected"]
            confidence = nn_result["confidence"]
            self.current_confidence = confidence

            self._print_status(sequence, confidence, detected)

            if detected != self.last_detection_state or (detected and confidence > 0.9):
                self._send_detection_to_esp32(detected, confidence)
                self.last_detection_state = detected

                if detected:
                    print(f"\n*** BEEP DETECTED! confidence={confidence:.3f} ***")
                    print(f"    Sent to ESP32 at {self.esp32_addr[0]}:{self.response_port}")

                    # Save detection for labeling ONLY if training mode is ON
                    if self.training_mode and len(self.continuous_buffer) >= self.sample_rate * 2:
                        audio_samples = np.array(
                            list(self.continuous_buffer)[-self.sample_rate * 2:],
                            dtype=np.int16
                        )
                        event = self.labeling_store.add_event(
                            confidence=confidence,
                            audio_samples=audio_samples,
                            sample_rate=self.sample_rate,
                        )
                        print(f"    Saved for labeling: {event.id}\n")
                    elif not self.training_mode:
                        print(f"    (Training mode OFF - not saving for labeling)\n")

        if self.recording:
            self.record_buffer.append(samples)

    def _print_status(self, seq: int, confidence: float, detected: bool):
        """Print real-time NN detection status."""
        if detected:
            color = "\033[92m"
        elif confidence > self.nn_detector.confidence_threshold * 0.7:
            color = "\033[93m"
        else:
            color = "\033[0m"

        bar_len = min(50, int(confidence * 50))
        bar = "#" * bar_len + "-" * (50 - bar_len)

        status = "BEEP!" if detected else "     "
        print(
            f"{color}seq={seq:8d} | conf={confidence:.3f} | [{bar}] {status}\033[0m",
            end="\r",
        )

    def _print_stats(self):
        """Print periodic statistics."""
        elapsed = time.time() - self.start_time
        pps = self.packets_received / elapsed if elapsed > 0 else 0
        kbps = (self.bytes_received * 8 / 1000) / elapsed if elapsed > 0 else 0
        loss_pct = (self.packets_lost / (self.packets_received + self.packets_lost) * 100) if self.packets_received > 0 else 0

        nn_rate = self.nn_inferences / elapsed if elapsed > 0 else 0

        print(f"\n[STATS] Packets: {self.packets_received}, Rate: {pps:.1f} pkt/s, {kbps:.1f} kbps, Loss: {loss_pct:.1f}%")
        print(f"        NN inferences: {self.nn_inferences}, Rate: {nn_rate:.1f}/s, Detections: {self.nn_detector.detection_count}")

        stats = self.labeling_store.get_stats()
        print(f"        Labeled: {stats['labeled']}/{stats['total_events']}, TP: {stats['true_positives']}, FP: {stats['false_positives']}\n")

    def _cleanup(self):
        """Clean up resources."""
        if self.recording:
            pass  # Could add stop_recording here

        self.sock.close()

        print(f"\nFinal stats:")
        print(f"  Packets received: {self.packets_received}")
        print(f"  Packets lost: {self.packets_lost}")
        print(f"  Bytes received: {self.bytes_received}")
        print(f"  NN inferences: {self.nn_inferences}")
        print(f"  Total detections: {self.nn_detector.detection_count}")


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
    high_confidence = [r for r in results if r["confidence"] > 0.8]

    print(f"\n" + "=" * 60)
    if high_confidence:
        print("SUCCESS: Beeps detected in audio file!")
    else:
        print("WARNING: No high-confidence beeps detected.")
    print("=" * 60)

    return 0 if high_confidence else 1


def main():
    parser = argparse.ArgumentParser(
        description="Audio Streaming Server for ESPHome Beep Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
    ESP32 --UDP audio--> Server (NN inference) --UDP detection--> ESP32 --ESPHome API--> Home Assistant

Examples:
    # Start server with web dashboard for training
    python audio_server.py --port 5050 --web-port 8080

    # Open dashboard at http://localhost:8080 to label detections
        """
    )
    parser.add_argument(
        "--port", type=int, default=5050, help="UDP port to receive audio (default: 5050)"
    )
    parser.add_argument(
        "--response-port", type=int, default=5001,
        help="UDP port to send detection results to ESP32 (default: 5001)"
    )
    parser.add_argument(
        "--web-port", type=int, default=8080,
        help="Port for web dashboard (default: 8080)"
    )
    parser.add_argument(
        "--sample-rate", type=int, default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--record-dir", type=str, default="recordings",
        help="Directory to save recordings (default: recordings)",
    )
    parser.add_argument(
        "--model-path", type=str, default="models/beep_detector.keras",
        help="Path to trained Keras model (default: models/beep_detector.keras)",
    )
    parser.add_argument(
        "--window-ms", type=int, default=500,
        help="Detection window size in ms (default: 500)",
    )
    parser.add_argument(
        "--confidence-threshold", type=float, default=0.7,
        help="Confidence threshold for NN detection (default: 0.7)",
    )
    parser.add_argument(
        "--test-file", type=str, default=None,
        help="Test detection on an audio file instead of live UDP stream",
    )

    args = parser.parse_args()

    if args.test_file:
        return test_on_file(
            audio_path=args.test_file,
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
        )

    server = AudioStreamServer(
        port=args.port,
        response_port=args.response_port,
        sample_rate=args.sample_rate,
        record_dir=args.record_dir,
        model_path=args.model_path,
        window_ms=args.window_ms,
        confidence_threshold=args.confidence_threshold,
        web_port=args.web_port,
    )

    server.start()


if __name__ == "__main__":
    exit(main() or 0)
