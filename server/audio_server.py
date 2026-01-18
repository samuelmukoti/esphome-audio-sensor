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
    pip install numpy scipy librosa flask
    # For inference only (no AVX required):
    pip install tflite-runtime
    # For full training support (requires AVX):
    pip install tensorflow
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

# Use TFLite runtime for inference (works on CPUs without AVX)
# Do NOT import TensorFlow here - it requires AVX instructions
TFLITE_AVAILABLE = False

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
    print("[INFO] Using TFLite runtime for inference (no AVX required)")
except ImportError as e:
    print(f"[WARNING] tflite-runtime not available: {e}")
    print("[WARNING] Model inference will be disabled")
    print("[WARNING] Install with: pip install tflite-runtime")

# Flask for web dashboard
try:
    from flask import Flask, render_template_string, jsonify, request, send_file
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Warning: Flask not installed. Web dashboard disabled. Install with: pip install flask")

# Ensemble Detector - imported conditionally when needed to avoid AVX requirement
# The import is deferred until use_ensemble=True to prevent TensorFlow from loading
# on CPUs without AVX support (like Intel Celeron N3050)
ENSEMBLE_AVAILABLE = False


# ============================================
# Detection Event Storage for Labeling
# ============================================

class DetectionEvent:
    """Stores a detection event with audio for labeling."""

    def __init__(self, event_id: str, timestamp: datetime, confidence: float,
                 audio_samples: np.ndarray, sample_rate: int,
                 source_type: str = "auto_detection",
                 detection_offset_ms: Optional[int] = None):
        self.id = event_id
        self.timestamp = timestamp
        self.confidence = confidence
        self.audio_samples = audio_samples
        self.sample_rate = sample_rate
        self.label: Optional[bool] = None  # None=unlabeled, True=beep, False=not beep
        self.labeled_at: Optional[datetime] = None
        # New fields for enhanced dashboard
        self.source_type = source_type  # "auto_detection" or "manual_capture"
        self.detection_offset_ms = detection_offset_ms  # Position of detection in clip
        self.notes: str = ""  # User notes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "duration_ms": len(self.audio_samples) / self.sample_rate * 1000 if len(self.audio_samples) > 0 else 0,
            "label": self.label,
            "labeled_at": self.labeled_at.isoformat() if self.labeled_at else None,
            "source_type": self.source_type,
            "detection_offset_ms": self.detection_offset_ms,
            "sample_rate": self.sample_rate,
            "notes": self.notes,
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

    SCHEMA_VERSION = 2  # Current schema version

    def __init__(self, data_dir: str = "labeled_data"):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        self.events: Dict[str, DetectionEvent] = {}
        self.max_events = 500  # Keep more events in memory for training
        self.training_history: List[dict] = []  # History of training runs
        # Configurable capture settings
        self.settings = {
            "auto_capture_enabled": True,
            "capture_duration_seconds": 5,
            "capture_confidence_min": 0.0,
            "capture_confidence_max": 1.0,
        }

        os.makedirs(self.audio_dir, exist_ok=True)
        self._load_labels()

    def _labels_file(self) -> str:
        return os.path.join(self.data_dir, "labels.json")

    def _load_labels(self):
        """Load existing labels from disk and restore events with schema migration."""
        labels_file = self._labels_file()
        if os.path.exists(labels_file):
            with open(labels_file, 'r') as f:
                data = json.load(f)

            # Schema migration: v1 -> v2
            needs_migration = 'version' not in data or data.get('version', 1) < self.SCHEMA_VERSION
            if needs_migration:
                print("[MIGRATION] Upgrading labels.json to schema v2...")
                data['version'] = self.SCHEMA_VERSION
                data.setdefault('training_history', [])
                data.setdefault('settings', {
                    'auto_capture_enabled': True,
                    'capture_duration_seconds': 5,
                    'capture_confidence_min': 0.0,
                    'capture_confidence_max': 1.0,
                })

                # Upgrade each sample
                for item in data.get('labeled', []):
                    # Infer source_type from confidence (0 = manual capture)
                    if item.get('confidence', 0) == 0:
                        item['source_type'] = 'manual_capture'
                        item['detection_offset_ms'] = None
                    else:
                        item['source_type'] = 'auto_detection'
                        # Estimate detection offset (end of clip minus detection window)
                        duration = item.get('duration_ms', 2000)
                        item['detection_offset_ms'] = max(0, int(duration - 500))

                    item.setdefault('sample_rate', 16000)
                    item.setdefault('notes', '')

            # Load training history and settings
            self.training_history = data.get('training_history', [])
            stored_settings = data.get('settings', {})
            self.settings.update(stored_settings)

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
                        sample_rate=item.get('sample_rate', 16000),
                        source_type=item.get('source_type', 'auto_detection'),
                        detection_offset_ms=item.get('detection_offset_ms'),
                    )
                    event.label = item['label']
                    event.notes = item.get('notes', '')
                    if item.get('labeled_at'):
                        event.labeled_at = datetime.fromisoformat(item['labeled_at'])

                    self.events[event_id] = event

            # Save migrated data
            if needs_migration:
                self._save_labels()
                print(f"[MIGRATION] Completed. Upgraded {len(self.events)} samples.")

            print(f"[LABELING] Restored {len(self.events)} labeled samples from disk")

    def _save_labels(self):
        """Save labels to disk with v2 schema."""
        labeled = [e.to_dict() for e in self.events.values() if e.label is not None]
        data = {
            "version": self.SCHEMA_VERSION,
            "labeled": labeled,
            "training_history": self.training_history,
            "settings": self.settings,
        }
        with open(self._labels_file(), 'w') as f:
            json.dump(data, f, indent=2)

    def add_event(self, confidence: float, audio_samples: np.ndarray,
                  sample_rate: int, source_type: str = "auto_detection",
                  detection_offset_ms: Optional[int] = None) -> DetectionEvent:
        """Add a new detection event."""
        event_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + str(uuid.uuid4())[:8]
        event = DetectionEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            confidence=confidence,
            audio_samples=audio_samples.copy(),
            sample_rate=sample_rate,
            source_type=source_type,
            detection_offset_ms=detection_offset_ms,
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

    def should_capture(self, confidence: float) -> bool:
        """Check if detection should be captured based on settings."""
        if not self.settings.get('auto_capture_enabled', True):
            return False
        conf_min = self.settings.get('capture_confidence_min', 0.0)
        conf_max = self.settings.get('capture_confidence_max', 1.0)
        return conf_min <= confidence <= conf_max

    def update_settings(self, new_settings: dict) -> dict:
        """Update capture settings and persist."""
        self.settings.update(new_settings)
        self._save_labels()
        return self.settings

    def add_training_run(self, samples_used: int, beeps: int, not_beeps: int,
                         epochs: int, final_accuracy: float, final_loss: float = 0.0) -> dict:
        """Record a training run in history."""
        run = {
            "timestamp": datetime.now().isoformat(),
            "samples_used": samples_used,
            "beeps": beeps,
            "not_beeps": not_beeps,
            "epochs": epochs,
            "final_accuracy": final_accuracy,
            "final_loss": final_loss,
        }
        self.training_history.append(run)
        self._save_labels()
        return run

    def get_samples(self, sample_type: str = "all", source: str = "all",
                    conf_min: float = 0.0, conf_max: float = 1.0,
                    page: int = 1, per_page: int = 20) -> dict:
        """Get filtered and paginated samples."""
        # Start with all events
        samples = list(self.events.values())

        # Filter by type
        if sample_type == "beep":
            samples = [s for s in samples if s.label is True]
        elif sample_type == "not_beep":
            samples = [s for s in samples if s.label is False]
        elif sample_type == "pending":
            samples = [s for s in samples if s.label is None]

        # Filter by source
        if source == "auto":
            samples = [s for s in samples if s.source_type == "auto_detection"]
        elif source == "manual":
            samples = [s for s in samples if s.source_type == "manual_capture"]

        # Filter by confidence range
        samples = [s for s in samples if conf_min <= s.confidence <= conf_max]

        # Sort by timestamp (newest first)
        samples = sorted(samples, key=lambda s: s.timestamp, reverse=True)

        # Paginate
        total = len(samples)
        start = (page - 1) * per_page
        end = start + per_page
        page_samples = samples[start:end]

        return {
            "samples": [s.to_dict() for s in page_samples],
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 0,
        }

    def update_sample(self, sample_id: str, label: Optional[bool] = None,
                      notes: Optional[str] = None) -> bool:
        """Update a sample's label or notes."""
        if sample_id not in self.events:
            return False

        event = self.events[sample_id]
        if label is not None:
            event.label = label
            event.labeled_at = datetime.now()
        if notes is not None:
            event.notes = notes

        self._save_labels()
        return True

    def batch_label(self, sample_ids: List[str], label: bool) -> int:
        """Label multiple samples at once."""
        count = 0
        for sample_id in sample_ids:
            if sample_id in self.events:
                self.events[sample_id].label = label
                self.events[sample_id].labeled_at = datetime.now()
                count += 1
        if count > 0:
            self._save_labels()
        return count

    def batch_delete(self, sample_ids: List[str]) -> int:
        """Delete multiple samples at once."""
        count = 0
        for sample_id in sample_ids:
            if sample_id in self.events:
                # Delete audio file
                audio_path = os.path.join(self.audio_dir, f"{sample_id}.wav")
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                del self.events[sample_id]
                count += 1
        if count > 0:
            self._save_labels()
        return count

    def get_waveform(self, sample_id: str, num_points: int = 500) -> Optional[dict]:
        """Get downsampled waveform data for visualization."""
        if sample_id not in self.events:
            return None

        audio_path = os.path.join(self.audio_dir, f"{sample_id}.wav")
        if not os.path.exists(audio_path):
            return None

        try:
            with wave.open(audio_path, 'rb') as wav:
                n_frames = wav.getnframes()
                sample_rate = wav.getframerate()
                audio_data = wav.readframes(n_frames)
                samples = np.frombuffer(audio_data, dtype=np.int16)

            # Downsample to num_points
            if len(samples) > num_points:
                indices = np.linspace(0, len(samples) - 1, num_points, dtype=int)
                downsampled = samples[indices]
            else:
                downsampled = samples

            # Normalize to -1 to 1
            normalized = downsampled.astype(float) / 32768.0

            event = self.events[sample_id]
            return {
                "waveform": normalized.tolist(),
                "duration_ms": len(samples) / sample_rate * 1000,
                "sample_rate": sample_rate,
                "detection_offset_ms": event.detection_offset_ms,
            }
        except Exception as e:
            print(f"[ERROR] Failed to load waveform for {sample_id}: {e}")
            return None

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

        # Record training run in history
        self.add_training_run(
            samples_used=len(X_new),
            beeps=int(n_pos),
            not_beeps=int(n_neg),
            epochs=epochs,
            final_accuracy=float(final_acc),
            final_loss=float(final_loss),
        )

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
        """Load the trained model (TFLite or Keras)."""
        self.model_path = model_path
        self.use_tflite = False
        self.tflite_interpreter = None
        self.tflite_input_details = None
        self.tflite_output_details = None

        # Try TFLite first (works on CPUs without AVX)
        tflite_path = model_path.replace('.keras', '.tflite')
        if TFLITE_AVAILABLE and os.path.exists(tflite_path):
            try:
                self.tflite_interpreter = tflite.Interpreter(model_path=tflite_path)
                self.tflite_interpreter.allocate_tensors()
                self.tflite_input_details = self.tflite_interpreter.get_input_details()
                self.tflite_output_details = self.tflite_interpreter.get_output_details()
                self.use_tflite = True
                self.model = True  # Flag that model is loaded
                print(f"  TFLite model loaded: {tflite_path}")
                return
            except Exception as e:
                print(f"  TFLite load failed: {e}")

        # TFLite is required - TensorFlow not supported due to AVX requirement
        print(f"  WARNING: TFLite model not found or failed to load")
        print(f"  TFLite runtime is required for legacy detector on this CPU")
        print(f"  Install with: pip install tflite-runtime")
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

        # Remove DC offset before normalization (critical for detection accuracy!)
        samples_centered = samples - np.mean(samples)
        y = samples_centered.astype(np.float32) / 32768.0
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

        mfcc_input = mfcc.reshape(1, expected_frames, self.n_mfcc).astype(np.float32)

        # Run inference using TFLite or Keras
        if self.use_tflite:
            self.tflite_interpreter.set_tensor(
                self.tflite_input_details[0]['index'], mfcc_input
            )
            self.tflite_interpreter.invoke()
            prediction = self.tflite_interpreter.get_tensor(
                self.tflite_output_details[0]['index']
            )[0][0]
        else:
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

        /* Navigation */
        .main-nav {
            display: flex;
            align-items: center;
            gap: 20px;
            padding: 15px 20px;
            background: #0f0f23;
            border-radius: 12px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        .nav-brand {
            font-size: 1.4em;
            font-weight: bold;
            color: #00d4ff;
        }
        .nav-tabs {
            display: flex;
            gap: 5px;
            flex: 1;
        }
        .nav-btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            background: #16213e;
            color: #aaa;
            cursor: pointer;
            font-size: 0.95em;
            transition: all 0.2s;
        }
        .nav-btn:hover { background: #1a3a5c; color: #fff; }
        .nav-btn.active {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000;
            font-weight: bold;
        }
        .audio-toggle-btn {
            padding: 8px 12px;
            border: none;
            border-radius: 8px;
            background: #16213e;
            color: #aaa;
            cursor: pointer;
            font-size: 1.1em;
            transition: all 0.2s;
            margin-left: 10px;
        }
        .audio-toggle-btn:hover { background: #1a3a5c; color: #fff; }
        .audio-toggle-btn.active {
            background: linear-gradient(135deg, #00ff88, #00cc66);
            color: #000;
            animation: pulse-audio 1.5s ease-in-out infinite;
        }
        @keyframes pulse-audio {
            0%, 100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.4); }
            50% { box-shadow: 0 0 0 8px rgba(0, 255, 136, 0); }
        }
        .nav-status {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 8px 15px;
            background: #16213e;
            border-radius: 20px;
            font-size: 0.85em;
        }
        .nav-status .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #666;
        }
        .nav-status .status-dot.connected { background: #00ff88; }

        /* Page containers */
        .page-content { display: none; }
        .page-content.active { display: block; }

        /* Settings panel */
        .settings-panel {
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .settings-row {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }
        .settings-row label { min-width: 150px; color: #aaa; }
        .settings-row input[type="range"] { flex: 1; min-width: 100px; }
        .settings-row input[type="checkbox"] { width: 20px; height: 20px; }
        .settings-row .value-display {
            min-width: 60px;
            text-align: right;
            color: #00d4ff;
            font-weight: bold;
        }

        /* Training page */
        .training-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .training-card {
            background: #16213e;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
        }
        .training-card .big-number {
            font-size: 3em;
            font-weight: bold;
            color: #00d4ff;
        }
        .training-card.positive .big-number { color: #00ff88; }
        .training-card.negative .big-number { color: #ff6b6b; }
        .training-history-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        .training-history-table th,
        .training-history-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        .training-history-table th { color: #888; font-weight: normal; }
        .training-history-table tr:hover { background: #1a3a5c; }

        /* Dataset management page */
        .filter-bar {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            align-items: center;
            background: #16213e;
            padding: 15px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .filter-bar select, .filter-bar input {
            padding: 8px 12px;
            border: 1px solid #333;
            border-radius: 6px;
            background: #0f0f23;
            color: #fff;
            font-size: 0.9em;
        }
        .filter-bar button {
            padding: 8px 16px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        }
        .btn-filter { background: #00d4ff; color: #000; }
        .btn-reset { background: #444; color: #fff; }
        .batch-actions {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }
        .batch-actions .selection-count { color: #888; margin-right: 10px; }

        /* Sample list with checkboxes */
        .sample-list-item {
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px;
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 10px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .sample-list-item:hover { background: #1a3a5c; }
        .sample-list-item.selected { background: #1a4a6c; border: 1px solid #00d4ff; }
        .sample-list-item input[type="checkbox"] { width: 18px; height: 18px; }
        .sample-list-item .sample-info { flex: 1; }
        .sample-list-item .sample-label {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .sample-list-item .sample-label.beep { background: #00ff88; color: #000; }
        .sample-list-item .sample-label.not-beep { background: #ff6b6b; color: #000; }
        .sample-list-item .sample-label.pending { background: #666; color: #fff; }

        /* Sample row (for dataset management) */
        .sample-row {
            display: grid;
            grid-template-columns: 30px 100px 100px 70px 70px 60px;
            align-items: center;
            gap: 15px;
            padding: 12px 15px;
            background: #16213e;
            border-radius: 8px;
            margin-bottom: 8px;
            transition: background 0.2s;
        }
        .sample-row:hover { background: #1a3a5c; }
        .sample-checkbox { width: 18px; height: 18px; cursor: pointer; }
        .sample-label {
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            text-align: center;
        }
        .sample-label.beep { background: #00ff88; color: #000; }
        .sample-label.not-beep { background: #ff6b6b; color: #000; }
        .sample-label.pending { background: #666; color: #fff; }
        .sample-date { color: #888; font-size: 0.85em; }
        .sample-confidence { color: #00d4ff; font-weight: bold; }
        .sample-source { color: #888; font-size: 0.85em; }
        .btn-view {
            padding: 6px 12px;
            border: none;
            border-radius: 4px;
            background: #0099cc;
            color: #fff;
            cursor: pointer;
            font-size: 0.85em;
        }
        .btn-view:hover { background: #00b8e8; }

        /* Sample list container */
        .sample-list {
            background: #0f0f23;
            border-radius: 10px;
            padding: 15px;
            max-height: 500px;
            overflow-y: auto;
        }

        /* Pagination */
        .pagination {
            display: flex;
            justify-content: center;
            gap: 5px;
            margin-top: 20px;
        }
        .pagination button {
            padding: 8px 14px;
            border: none;
            border-radius: 6px;
            background: #16213e;
            color: #fff;
            cursor: pointer;
        }
        .pagination button:hover { background: #1a3a5c; }
        .pagination button.active { background: #00d4ff; color: #000; }
        .pagination button:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Loading states */
        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: #888;
        }
        .loading::after {
            content: '';
            width: 24px;
            height: 24px;
            border: 3px solid #333;
            border-top-color: #00d4ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-left: 10px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .btn-loading {
            position: relative;
            pointer-events: none;
            opacity: 0.7;
        }
        .btn-loading::after {
            content: '';
            position: absolute;
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top-color: currentColor;
            border-radius: 50%;
            animation: spin 0.6s linear infinite;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .sample-row {
                grid-template-columns: 30px 1fr 60px;
                gap: 10px;
            }
            .sample-row .sample-date,
            .sample-row .sample-source { display: none; }
            .training-stats {
                grid-template-columns: repeat(2, 1fr);
            }
            .filter-bar {
                flex-direction: column;
                align-items: stretch;
            }
            .batch-actions {
                flex-wrap: wrap;
            }
            .main-nav {
                flex-direction: column;
                align-items: stretch;
            }
            .nav-tabs {
                justify-content: center;
            }
        }

        /* Sample detail modal */
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        .modal-content {
            background: #16213e;
            padding: 30px;
            border-radius: 15px;
            max-width: 700px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
        }
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .modal-close {
            font-size: 1.5em;
            cursor: pointer;
            color: #888;
        }
        .modal-close:hover { color: #fff; }
        .waveform-container {
            position: relative;
            background: #0a0a1a;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
        }
        .waveform-container canvas { width: 100%; height: 120px; }
        .detection-marker {
            position: absolute;
            top: 10px;
            bottom: 10px;
            width: 2px;
            background: #ff6b6b;
        }
        .playback-controls {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 20px;
        }
        .playback-controls input[type="range"] { flex: 1; }
        .label-buttons {
            display: flex;
            gap: 15px;
            margin-bottom: 20px;
        }
        .label-buttons button {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
        }
        .notes-field textarea {
            width: 100%;
            padding: 12px;
            border: 1px solid #333;
            border-radius: 8px;
            background: #0a0a1a;
            color: #fff;
            resize: vertical;
            min-height: 80px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Navigation Bar -->
        <nav class="main-nav">
            <div class="nav-brand">🔊 Beep Detector</div>
            <div class="nav-tabs">
                <button class="nav-btn active" data-page="monitor" onclick="showPage('monitor')">
                    📊 Monitor
                </button>
                <button class="nav-btn" data-page="training" onclick="showPage('training')">
                    🧠 Training
                </button>
                <button class="nav-btn" data-page="dataset" onclick="showPage('dataset')">
                    📁 Dataset
                </button>
            </div>
            <button class="audio-toggle-btn" id="audio-toggle-btn" onclick="toggleLiveAudio()" title="Toggle live audio playback">
                🔇
            </button>
            <div class="nav-status">
                <div class="status-dot" id="nav-status-dot"></div>
                <span id="nav-status-text">Connecting...</span>
            </div>
        </nav>

        <!-- Page: Monitor & Analytics -->
        <div id="page-monitor" class="page-content active">

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
            <h2>📚 Quick Dataset View</h2>
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
            <p style="color: #888; margin-top: 10px;">Go to <a href="#" onclick="showPage('dataset'); return false;" style="color: #00d4ff;">Dataset</a> tab for full management.</p>
        </div>

        </div><!-- End page-monitor -->

        <!-- Page: Training -->
        <div id="page-training" class="page-content">
            <h2 style="color: #00d4ff; margin-bottom: 20px;">🧠 Model Training</h2>

            <!-- Capture Settings -->
            <div class="settings-panel">
                <h3 style="margin-bottom: 15px;">⚙️ Auto-Capture Settings</h3>
                <div class="settings-row">
                    <label>Auto-capture enabled:</label>
                    <input type="checkbox" id="auto-capture-enabled" checked onchange="updateSettings()">
                    <span style="color: #888;">Automatically save detected audio for labeling</span>
                </div>
                <div class="settings-row">
                    <label>Capture duration:</label>
                    <select id="capture-duration" onchange="updateSettings()" style="padding: 8px; background: #2a2a4e; color: #fff; border: 1px solid #444; border-radius: 4px;">
                        <option value="3">3 seconds</option>
                        <option value="5" selected>5 seconds</option>
                        <option value="7">7 seconds</option>
                        <option value="10">10 seconds</option>
                    </select>
                </div>
                <div class="settings-row">
                    <label>Confidence range:</label>
                    <input type="range" id="conf-min" min="0" max="1" step="0.05" value="0" oninput="updateConfDisplay('min')">
                    <span id="conf-min-val">0.00</span>
                    <span style="margin: 0 10px;">to</span>
                    <input type="range" id="conf-max" min="0" max="1" step="0.05" value="1" oninput="updateConfDisplay('max')">
                    <span id="conf-max-val">1.00</span>
                    <button onclick="updateSettings()" style="margin-left: 15px; padding: 6px 12px; background: #00d4ff; color: #000; border: none; border-radius: 4px; cursor: pointer;">Save</button>
                </div>
                <p style="color: #666; font-size: 0.85em; margin-top: 10px;">
                    Only captures detections with confidence between min and max values. Use lower range (0.5-0.85) to focus on borderline cases.
                </p>
            </div>

            <!-- Dataset Summary -->
            <div class="training-stats">
                <div class="training-card positive">
                    <div class="big-number" id="summary-beep-count">0</div>
                    <div style="color: #888;">Beep Samples</div>
                </div>
                <div class="training-card negative">
                    <div class="big-number" id="summary-not-beep-count">0</div>
                    <div style="color: #888;">Not Beep Samples</div>
                </div>
                <div class="training-card">
                    <div class="big-number" id="summary-pending-count">0</div>
                    <div style="color: #888;">Pending Review</div>
                </div>
                <div class="training-card">
                    <div class="big-number" id="summary-total-count">0</div>
                    <div style="color: #888;">Total Labeled</div>
                </div>
            </div>

            <!-- Training Controls -->
            <div class="settings-panel training-controls">
                <h3 style="margin-bottom: 15px;">🚀 Train Model</h3>
                <div style="display: flex; gap: 20px; align-items: center; flex-wrap: wrap; margin-bottom: 15px;">
                    <div>
                        <label style="color: #888; font-size: 0.9em;">Epochs:</label>
                        <select id="training-epochs" style="padding: 8px; background: #2a2a4e; color: #fff; border: 1px solid #444; border-radius: 4px; margin-left: 5px;">
                            <option value="10">10</option>
                            <option value="20" selected>20</option>
                            <option value="30">30</option>
                            <option value="50">50</option>
                        </select>
                    </div>
                    <div>
                        <label style="color: #888; font-size: 0.9em;">Learning Rate:</label>
                        <select id="learning-rate" style="padding: 8px; background: #2a2a4e; color: #fff; border: 1px solid #444; border-radius: 4px; margin-left: 5px;">
                            <option value="0.0001">0.0001</option>
                            <option value="0.001" selected>0.001</option>
                            <option value="0.01">0.01</option>
                        </select>
                    </div>
                </div>
                <button class="btn-primary" onclick="startTrainingFromPage()" style="padding: 15px 30px; font-size: 1.1em; background: linear-gradient(135deg, #00d4ff, #00ff88); color: #000; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;">
                    🧠 Retrain Model Now
                </button>
                <p style="color: #666; margin-top: 10px; font-size: 0.9em;">
                    Uses all labeled samples to fine-tune the neural network model.
                </p>
            </div>

            <!-- Training History -->
            <div class="settings-panel">
                <h3 style="margin-bottom: 15px;">📜 Training History</h3>
                <div id="training-history-container">
                    <table class="training-history-table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Samples</th>
                                <th>Beeps</th>
                                <th>Not Beeps</th>
                                <th>Epochs</th>
                                <th>Accuracy</th>
                            </tr>
                        </thead>
                        <tbody id="training-history-body">
                            <tr><td colspan="6" style="color: #666; text-align: center;">No training runs yet</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div><!-- End page-training -->

        <!-- Page: Dataset Management -->
        <div id="page-dataset" class="page-content">
            <h2 style="color: #00d4ff; margin-bottom: 20px;">📁 Dataset Management</h2>

            <!-- Filter Bar -->
            <div class="filter-bar">
                <select id="filter-type">
                    <option value="all">All Types</option>
                    <option value="beep">Beep Only</option>
                    <option value="not_beep">Not Beep Only</option>
                    <option value="pending">Pending Review</option>
                </select>
                <select id="filter-source">
                    <option value="all">All Sources</option>
                    <option value="auto">Auto-Detected</option>
                    <option value="manual">Manual Capture</option>
                </select>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="color: #888;">Confidence:</span>
                    <input type="number" id="filter-conf-min" min="0" max="100" value="0" style="width: 60px;">
                    <span>-</span>
                    <input type="number" id="filter-conf-max" min="0" max="100" value="100" style="width: 60px;">
                    <span style="color: #888;">%</span>
                </div>
                <button class="btn-filter" onclick="applyFilters()">Apply</button>
                <button class="btn-reset" onclick="resetFilters()">Reset</button>
            </div>

            <!-- Batch Actions -->
            <div class="batch-actions">
                <span class="selection-count"><span id="selected-count">0</span> selected</span>
                <button class="btn-correct" onclick="batchLabel(true)" style="padding: 8px 16px;">✓ Label as Beep</button>
                <button class="btn-incorrect" onclick="batchLabel(false)" style="padding: 8px 16px;">✗ Label as Not Beep</button>
                <button class="btn-delete" onclick="batchDelete()">🗑️ Delete Selected</button>
                <button id="select-all-btn" style="background: #444; color: #fff; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;" onclick="selectAllSamples()">Select All</button>
                <button id="select-none-btn" style="background: #333; color: #888; padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;" onclick="selectNoneSamples()">Clear</button>
            </div>

            <!-- Sample List -->
            <div id="dataset-sample-list" class="sample-list"></div>

            <!-- Pagination -->
            <div class="pagination">
                <button id="prev-page" onclick="prevPage()" style="padding: 8px 16px; background: #2a2a4e; color: #fff; border: 1px solid #444; border-radius: 6px; cursor: pointer;">← Prev</button>
                <span id="page-info" style="color: #888; margin: 0 15px;">Page 1 of 1</span>
                <button id="next-page" onclick="nextPage()" style="padding: 8px 16px; background: #2a2a4e; color: #fff; border: 1px solid #444; border-radius: 6px; cursor: pointer;">Next →</button>
            </div>
        </div><!-- End page-dataset -->

        <!-- Sample Detail Modal -->
        <div id="sample-modal" class="modal" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Sample: <span id="modal-sample-id"></span></h3>
                    <span class="modal-close" onclick="closeSampleModal()">&times;</span>
                </div>

                <div class="waveform-container">
                    <canvas id="waveform-canvas" width="640" height="120"></canvas>
                </div>

                <div class="playback-controls">
                    <button onclick="playModalAudio()" style="padding: 8px 16px; background: #00d4ff; color: #000; border: none; border-radius: 6px; cursor: pointer;">▶ Play</button>
                    <button onclick="pauseModalAudio()" style="padding: 8px 16px; background: #444; color: #fff; border: none; border-radius: 6px; cursor: pointer;">⏸ Pause</button>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; color: #888;">
                    <div><strong>Timestamp:</strong> <span id="modal-timestamp"></span></div>
                    <div><strong>Confidence:</strong> <span id="modal-confidence"></span></div>
                    <div><strong>Source:</strong> <span id="modal-source"></span></div>
                    <div><strong>Status:</strong> <span id="modal-label-status"></span></div>
                </div>

                <div class="label-buttons" style="margin-bottom: 20px;">
                    <button class="btn-correct" onclick="labelFromModal(true)" style="padding: 12px 24px;">✓ Label as Beep</button>
                    <button class="btn-incorrect" onclick="labelFromModal(false)" style="padding: 12px 24px;">✗ Label as Not Beep</button>
                </div>

                <div class="notes-field">
                    <label style="color: #888; display: block; margin-bottom: 5px;">Notes:</label>
                    <textarea id="modal-notes" placeholder="Add notes about this sample..."></textarea>
                    <button onclick="saveModalNotes()" style="margin-top: 10px; padding: 8px 16px; background: #00d4ff; color: #000; border: none; border-radius: 6px; cursor: pointer;">Save Notes</button>
                </div>

                <button onclick="deleteFromModal()" style="margin-top: 15px; padding: 10px 20px; background: #8b0000; color: #fff; border: none; border-radius: 6px; cursor: pointer; width: 100%;">🗑️ Delete Sample</button>
            </div>
        </div>

    </div><!-- End container -->

    <script>
        // Live audio playback state
        let audioContext = null;
        let gainNode = null;
        let isLiveAudioEnabled = false;
        let audioFetchInterval = null;
        let nextPlayTime = 0;

        // Buffered playback state
        let audioQueue = [];
        let isBuffering = true;
        let audioReadCursor = null;
        const PRE_BUFFER_MS = 500;  // Buffer 500ms before starting playback
        const CHUNK_MS = 100;  // Each chunk is ~100ms

        function toggleLiveAudio() {
            isLiveAudioEnabled = !isLiveAudioEnabled;
            const btn = document.getElementById('audio-toggle-btn');

            if (isLiveAudioEnabled) {
                // Pause spectrum visualization to save resources
                pauseSpectrumVisualization();
                startLiveAudioPlayback();
                btn.textContent = '🔊';
                btn.classList.add('active');
                btn.title = 'Live audio ON - click to mute (spectrum paused)';
            } else {
                stopLiveAudioPlayback();
                // Resume spectrum visualization
                resumeSpectrumVisualization();
                btn.textContent = '🔇';
                btn.classList.remove('active');
                btn.title = 'Toggle live audio playback';
            }
        }

        async function startLiveAudioPlayback() {
            try {
                // Create audio context with browser's native sample rate
                // Web Audio API will resample from 16kHz source automatically
                audioContext = new (window.AudioContext || window.webkitAudioContext)();

                // Resume context if suspended (required by browser autoplay policy)
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }
                console.log('AudioContext state:', audioContext.state, 'native sampleRate:', audioContext.sampleRate);

                // Create gain node to boost volume (mic audio can be quiet)
                gainNode = audioContext.createGain();
                gainNode.gain.value = 3.0; // Boost volume 3x
                gainNode.connect(audioContext.destination);

                nextPlayTime = audioContext.currentTime;

                // Reset buffering state
                audioQueue = [];
                isBuffering = true;
                audioReadCursor = null;  // Will be set by first server response
                updateBufferingIndicator(true);

                // Start fetching audio chunks
                audioFetchInterval = setInterval(fetchAndPlayAudioChunk, 100);
                console.log('Live audio playback started (buffering...)');
            } catch (err) {
                console.error('Failed to start audio playback:', err);
                isLiveAudioEnabled = false;
                document.getElementById('audio-toggle-btn').textContent = '🔇';
                document.getElementById('audio-toggle-btn').classList.remove('active');
                updateBufferingIndicator(false);
            }
        }

        function updateBufferingIndicator(buffering) {
            const btn = document.getElementById('audio-toggle-btn');
            if (buffering && isLiveAudioEnabled) {
                btn.textContent = '⏳';
                btn.title = 'Buffering audio...';
            } else if (isLiveAudioEnabled) {
                btn.textContent = '🔊';
                btn.title = 'Live audio ON - click to mute';
            }
        }

        function stopLiveAudioPlayback() {
            if (audioFetchInterval) {
                clearInterval(audioFetchInterval);
                audioFetchInterval = null;
            }
            if (gainNode) {
                gainNode.disconnect();
                gainNode = null;
            }
            if (audioContext) {
                audioContext.close();
                audioContext = null;
            }
            nextPlayTime = 0;
            // Reset buffering state
            audioQueue = [];
            isBuffering = true;
            audioReadCursor = null;
            updateBufferingIndicator(false);
            console.log('Live audio playback stopped');
        }

        async function fetchAndPlayAudioChunk() {
            if (!audioContext || !isLiveAudioEnabled) return;

            try {
                // Build URL with cursor for sequential reads
                let url = '/api/audio-stream';
                if (audioReadCursor !== null) {
                    url += '?cursor=' + audioReadCursor;
                }

                const response = await fetch(url);
                const data = await response.json();

                if (data.audio) {
                    // Update cursor for next fetch
                    if (data.next_cursor !== undefined) {
                        audioReadCursor = data.next_cursor;
                    }

                    // Decode base64 to ArrayBuffer
                    const binaryString = atob(data.audio);
                    const bytes = new Uint8Array(binaryString.length);
                    for (let i = 0; i < binaryString.length; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }

                    // Convert int16 to float32 for Web Audio API
                    const int16Array = new Int16Array(bytes.buffer);
                    const float32Array = new Float32Array(int16Array.length);
                    for (let i = 0; i < int16Array.length; i++) {
                        float32Array[i] = int16Array[i] / 32768.0;
                    }

                    // Create audio buffer
                    const audioBuffer = audioContext.createBuffer(1, float32Array.length, data.sample_rate);
                    audioBuffer.getChannelData(0).set(float32Array);

                    // If still buffering, queue the chunk
                    if (isBuffering) {
                        audioQueue.push(audioBuffer);
                        const bufferedMs = audioQueue.length * CHUNK_MS;
                        console.log(`Buffering: ${bufferedMs}ms / ${PRE_BUFFER_MS}ms`);

                        // Check if we have enough buffered
                        if (bufferedMs >= PRE_BUFFER_MS) {
                            isBuffering = false;
                            updateBufferingIndicator(false);
                            console.log('Buffer ready, starting playback');
                            // Play all queued audio
                            nextPlayTime = audioContext.currentTime;
                            while (audioQueue.length > 0) {
                                const buf = audioQueue.shift();
                                scheduleAudioBuffer(buf);
                            }
                        }
                        return;
                    }

                    // Normal playback - schedule immediately
                    scheduleAudioBuffer(audioBuffer);
                }
            } catch (err) {
                // Silently ignore fetch errors (may happen if server is busy)
                console.debug('Audio fetch error:', err);
            }
        }

        function scheduleAudioBuffer(audioBuffer) {
            if (!audioContext || !gainNode) return;

            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(gainNode);

            // Schedule to play at next available time
            const now = audioContext.currentTime;
            if (nextPlayTime < now) {
                // If we've fallen behind, reset to now (will cause a gap but prevent buildup)
                if (now - nextPlayTime > 0.5) {
                    console.log('Audio playback fell behind, resetting');
                }
                nextPlayTime = now;
            }
            source.start(nextPlayTime);
            nextPlayTime += audioBuffer.duration;
        }

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

        // Update spectrum at ~20 FPS (stored so we can pause during audio playback)
        let spectrumIntervalId = setInterval(fetchSpectrum, 50);

        function pauseSpectrumVisualization() {
            if (spectrumIntervalId) {
                clearInterval(spectrumIntervalId);
                spectrumIntervalId = null;
                console.log('Spectrum visualization paused');
            }
        }

        function resumeSpectrumVisualization() {
            if (!spectrumIntervalId) {
                spectrumIntervalId = setInterval(fetchSpectrum, 50);
                console.log('Spectrum visualization resumed');
            }
        }

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

        // ========================================
        // Multi-Page Navigation
        // ========================================
        let currentPage = 'monitor';

        function showPage(pageName) {
            currentPage = pageName;

            // Hide all pages
            document.querySelectorAll('.page-content').forEach(page => {
                page.classList.remove('active');
            });

            // Show selected page
            const targetPage = document.getElementById('page-' + pageName);
            if (targetPage) {
                targetPage.classList.add('active');
            }

            // Update nav tabs
            document.querySelectorAll('.nav-btn').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelector(`.nav-btn[onclick*="${pageName}"]`).classList.add('active');

            // Load page-specific data
            if (pageName === 'training') {
                loadSettings();
                fetchTrainingHistory();
                updateDatasetSummary();
            } else if (pageName === 'dataset') {
                loadDatasetSamples();
            }
        }

        // ========================================
        // Settings Management
        // ========================================
        function loadSettings() {
            fetch('/api/settings')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('auto-capture-enabled').checked = data.auto_capture_enabled !== false;
                    document.getElementById('capture-duration').value = data.capture_duration_seconds || 5;
                    document.getElementById('conf-min').value = data.capture_confidence_min || 0;
                    document.getElementById('conf-max').value = data.capture_confidence_max || 1;

                    // Update display values
                    document.getElementById('conf-min-val').textContent = (data.capture_confidence_min || 0).toFixed(2);
                    document.getElementById('conf-max-val').textContent = (data.capture_confidence_max || 1).toFixed(2);
                })
                .catch(err => {
                    console.error('Failed to load settings:', err);
                    showToast('Failed to load settings', 'error');
                });
        }

        function updateSettings() {
            const settings = {
                auto_capture_enabled: document.getElementById('auto-capture-enabled').checked,
                capture_duration_seconds: parseInt(document.getElementById('capture-duration').value),
                capture_confidence_min: parseFloat(document.getElementById('conf-min').value),
                capture_confidence_max: parseFloat(document.getElementById('conf-max').value)
            };

            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast('Settings saved!', 'success');
                } else {
                    showToast('Failed to save settings', 'error');
                }
            });
        }

        function updateConfDisplay(type) {
            const slider = document.getElementById('conf-' + type);
            document.getElementById('conf-' + type + '-val').textContent = parseFloat(slider.value).toFixed(2);
        }

        // ========================================
        // Training Page Functions
        // ========================================
        function updateDatasetSummary() {
            fetch('/api/dataset')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('summary-beep-count').textContent = data.stats.positive;
                    document.getElementById('summary-not-beep-count').textContent = data.stats.negative;
                    document.getElementById('summary-total-count').textContent = data.stats.total;

                    // Calculate unlabeled from events
                    fetch('/api/events')
                        .then(r => r.json())
                        .then(evtData => {
                            document.getElementById('summary-pending-count').textContent = evtData.events.length;
                        });
                });
        }

        function fetchTrainingHistory() {
            fetch('/api/training-history')
                .then(r => r.json())
                .then(data => {
                    const tbody = document.getElementById('training-history-body');

                    if (!data.history || data.history.length === 0) {
                        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:#666;padding:20px;">No training runs yet</td></tr>';
                        return;
                    }

                    tbody.innerHTML = data.history.map(run => {
                        const date = new Date(run.timestamp).toLocaleString();
                        const accuracy = (run.final_accuracy * 100).toFixed(1) + '%';
                        return `
                            <tr>
                                <td>${date}</td>
                                <td>${run.samples_used}</td>
                                <td>${run.beeps}</td>
                                <td>${run.not_beeps}</td>
                                <td>${run.epochs}</td>
                                <td style="color: ${run.final_accuracy > 0.9 ? '#00ff88' : '#ffaa00'}">${accuracy}</td>
                            </tr>
                        `;
                    }).join('');
                })
                .catch(err => {
                    console.error('Failed to fetch training history:', err);
                });
        }

        function startTrainingFromPage() {
            const epochs = parseInt(document.getElementById('training-epochs').value);
            const learningRate = parseFloat(document.getElementById('learning-rate').value);

            const btn = document.querySelector('.training-controls .btn-primary');
            const originalText = btn.textContent;
            btn.textContent = '⏳ Training...';
            btn.disabled = true;

            fetch('/api/retrain', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({epochs: epochs, learning_rate: learningRate})
            })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        btn.textContent = '✓ Complete!';
                        btn.style.background = '#00ff88';
                        showToast(`Model trained! Accuracy: ${(data.final_accuracy * 100).toFixed(1)}%`, 'success');
                        fetchTrainingHistory();
                        updateDatasetSummary();
                    } else {
                        btn.textContent = '❌ Failed';
                        showToast('Training failed: ' + (data.error || 'Unknown error'), 'error');
                    }
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.style.background = '';
                        btn.disabled = false;
                    }, 3000);
                })
                .catch(err => {
                    btn.textContent = '❌ Error';
                    showToast('Training error: ' + err, 'error');
                    setTimeout(() => {
                        btn.textContent = originalText;
                        btn.disabled = false;
                    }, 3000);
                });
        }

        // ========================================
        // Dataset Management Page Functions
        // ========================================
        let datasetPage = 1;
        let datasetTotalPages = 1;
        let datasetItems = [];
        let selectedSamples = new Set();

        function loadDatasetSamples() {
            const typeFilter = document.getElementById('filter-type').value;
            const sourceFilter = document.getElementById('filter-source').value;
            const confMin = parseFloat(document.getElementById('filter-conf-min').value) / 100;
            const confMax = parseFloat(document.getElementById('filter-conf-max').value) / 100;

            let url = `/api/samples?page=${datasetPage}&per_page=20`;
            if (typeFilter && typeFilter !== 'all') url += `&type=${typeFilter}`;
            if (sourceFilter && sourceFilter !== 'all') url += `&source=${sourceFilter}`;
            if (confMin > 0) url += `&conf_min=${confMin}`;
            if (confMax < 1) url += `&conf_max=${confMax}`;

            // Show loading state
            const container = document.getElementById('dataset-sample-list');
            container.innerHTML = '<div class="loading">Loading samples</div>';

            fetch(url)
                .then(r => r.json())
                .then(data => {
                    datasetItems = data.samples || [];
                    datasetTotalPages = data.total_pages || 1;
                    selectedSamples.clear();
                    renderDatasetPage();
                    updatePagination();
                    updateBatchButtons();
                })
                .catch(err => {
                    container.innerHTML = '<div style="text-align:center;color:#ff6b6b;padding:40px;">Failed to load samples. Please try again.</div>';
                    showToast('Failed to load samples: ' + err, 'error');
                });
        }

        function renderDatasetPage() {
            const container = document.getElementById('dataset-sample-list');

            if (datasetItems.length === 0) {
                container.innerHTML = '<div style="text-align:center;color:#666;padding:40px;">No samples match your filters</div>';
                return;
            }

            container.innerHTML = datasetItems.map(s => {
                const labelText = s.label === null ? '⏳ Pending' : (s.label ? '✓ Beep' : '✗ Not Beep');
                const labelClass = s.label === null ? 'pending' : (s.label ? 'beep' : 'not-beep');
                const sourceText = s.source_type === 'manual_capture' ? 'Manual' : 'Auto';
                const date = new Date(s.timestamp).toLocaleDateString();
                const checked = selectedSamples.has(s.id) ? 'checked' : '';

                return `
                    <div class="sample-row" data-id="${s.id}">
                        <input type="checkbox" class="sample-checkbox" ${checked} onchange="toggleSampleSelection('${s.id}')">
                        <span class="sample-label ${labelClass}">${labelText}</span>
                        <span class="sample-date">${date}</span>
                        <span class="sample-confidence">${(s.confidence * 100).toFixed(0)}%</span>
                        <span class="sample-source">${sourceText}</span>
                        <button class="btn-view" onclick="openSampleModal('${s.id}')">View</button>
                    </div>
                `;
            }).join('');
        }

        function updatePagination() {
            document.getElementById('page-info').textContent = `Page ${datasetPage} of ${datasetTotalPages}`;
            document.getElementById('prev-page').disabled = datasetPage <= 1;
            document.getElementById('next-page').disabled = datasetPage >= datasetTotalPages;
        }

        function prevPage() {
            if (datasetPage > 1) {
                datasetPage--;
                loadDatasetSamples();
            }
        }

        function nextPage() {
            if (datasetPage < datasetTotalPages) {
                datasetPage++;
                loadDatasetSamples();
            }
        }

        function applyFilters() {
            datasetPage = 1;
            loadDatasetSamples();
        }

        function resetFilters() {
            document.getElementById('filter-type').value = '';
            document.getElementById('filter-source').value = '';
            document.getElementById('filter-conf-min').value = '';
            document.getElementById('filter-conf-max').value = '';
            datasetPage = 1;
            loadDatasetSamples();
        }

        function toggleSampleSelection(id) {
            if (selectedSamples.has(id)) {
                selectedSamples.delete(id);
            } else {
                selectedSamples.add(id);
            }
            updateBatchButtons();
        }

        function selectAllSamples() {
            datasetItems.forEach(s => selectedSamples.add(s.id));
            renderDatasetPage();
            updateBatchButtons();
        }

        function selectNoneSamples() {
            selectedSamples.clear();
            renderDatasetPage();
            updateBatchButtons();
        }

        function updateBatchButtons() {
            const count = selectedSamples.size;
            document.getElementById('selected-count').textContent = count;
            document.querySelectorAll('.batch-actions button').forEach(btn => {
                if (btn.id !== 'select-all-btn' && btn.id !== 'select-none-btn') {
                    btn.disabled = count === 0;
                }
            });
        }

        function batchLabel(isBeep) {
            if (selectedSamples.size === 0) return;

            const label = isBeep ? 'beep' : 'not-beep';
            if (!confirm(`Label ${selectedSamples.size} samples as "${label}"?`)) return;

            fetch('/api/samples/batch-label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ids: Array.from(selectedSamples), label: isBeep})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast(`Labeled ${data.updated} samples`, 'success');
                    loadDatasetSamples();
                    fetchEvents();
                } else {
                    showToast('Batch label failed: ' + data.error, 'error');
                }
            });
        }

        function batchDelete() {
            if (selectedSamples.size === 0) return;

            if (!confirm(`Delete ${selectedSamples.size} samples? This cannot be undone.`)) return;

            fetch('/api/samples/batch-delete', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ids: Array.from(selectedSamples)})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast(`Deleted ${data.deleted} samples`, 'success');
                    loadDatasetSamples();
                    fetchEvents();
                } else {
                    showToast('Batch delete failed: ' + data.error, 'error');
                }
            });
        }

        // ========================================
        // Sample Detail Modal
        // ========================================
        let currentModalSample = null;
        let modalAudio = null;

        function openSampleModal(id) {
            currentModalSample = id;
            const modal = document.getElementById('sample-modal');
            modal.style.display = 'flex';

            // Load sample details
            const sample = datasetItems.find(s => s.id === id) || {};

            document.getElementById('modal-sample-id').textContent = id;
            document.getElementById('modal-timestamp').textContent = new Date(sample.timestamp).toLocaleString();
            document.getElementById('modal-confidence').textContent = (sample.confidence * 100).toFixed(1) + '%';
            document.getElementById('modal-source').textContent = sample.source_type === 'manual_capture' ? 'Manual Capture' : 'Auto Detection';
            document.getElementById('modal-notes').value = sample.notes || '';

            // Update label buttons state
            const labelStatus = document.getElementById('modal-label-status');
            if (sample.label === null) {
                labelStatus.textContent = 'Pending';
                labelStatus.style.color = '#888';
            } else if (sample.label) {
                labelStatus.textContent = 'Labeled: Beep';
                labelStatus.style.color = '#00ff88';
            } else {
                labelStatus.textContent = 'Labeled: Not Beep';
                labelStatus.style.color = '#ff6b6b';
            }

            // Load waveform
            loadWaveform(id, sample.detection_offset_ms, sample.duration_ms || 5000);

            // Setup audio
            modalAudio = new Audio(`/api/audio/${id}`);
        }

        function closeSampleModal() {
            document.getElementById('sample-modal').style.display = 'none';
            currentModalSample = null;
            if (modalAudio) {
                modalAudio.pause();
                modalAudio = null;
            }
        }

        function loadWaveform(id, detectionOffsetMs, durationMs) {
            fetch(`/api/samples/${id}/waveform`)
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        drawWaveform(data.waveform, detectionOffsetMs, durationMs);
                    }
                });
        }

        function drawWaveform(waveformData, detectionOffsetMs, durationMs) {
            const canvas = document.getElementById('waveform-canvas');
            const ctx = canvas.getContext('2d');
            const width = canvas.width;
            const height = canvas.height;

            // Clear
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(0, 0, width, height);

            if (!waveformData || waveformData.length === 0) return;

            // Draw waveform
            ctx.strokeStyle = '#4ecdc4';
            ctx.lineWidth = 1;
            ctx.beginPath();

            const maxVal = Math.max(...waveformData.map(Math.abs)) || 1;
            waveformData.forEach((val, i) => {
                const x = (i / waveformData.length) * width;
                const y = height/2 - (val / maxVal * height * 0.4);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            });
            ctx.stroke();

            // Draw detection marker
            if (detectionOffsetMs && durationMs) {
                const markerX = (detectionOffsetMs / durationMs) * width;
                ctx.strokeStyle = '#ff6b6b';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(markerX, 0);
                ctx.lineTo(markerX, height);
                ctx.stroke();

                // Label
                ctx.fillStyle = '#ff6b6b';
                ctx.font = '10px monospace';
                ctx.fillText('Detection', markerX + 5, 15);
            }

            // Time axis labels
            ctx.fillStyle = '#666';
            ctx.font = '10px monospace';
            const duration = durationMs / 1000;
            for (let t = 0; t <= duration; t++) {
                const x = (t / duration) * width;
                ctx.fillText(t + 's', x, height - 5);
            }
        }

        function playModalAudio() {
            if (modalAudio) {
                modalAudio.currentTime = 0;
                modalAudio.play();
            }
        }

        function pauseModalAudio() {
            if (modalAudio) {
                modalAudio.pause();
            }
        }

        function labelFromModal(isBeep) {
            if (!currentModalSample) return;

            fetch(`/api/samples/${currentModalSample}`, {
                method: 'PATCH',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({label: isBeep})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast('Sample labeled!', 'success');

                    // Update modal display
                    const labelStatus = document.getElementById('modal-label-status');
                    if (isBeep) {
                        labelStatus.textContent = 'Labeled: Beep';
                        labelStatus.style.color = '#00ff88';
                    } else {
                        labelStatus.textContent = 'Labeled: Not Beep';
                        labelStatus.style.color = '#ff6b6b';
                    }

                    // Refresh lists
                    loadDatasetSamples();
                    fetchEvents();
                } else {
                    showToast('Failed to label: ' + data.error, 'error');
                }
            });
        }

        function saveModalNotes() {
            if (!currentModalSample) return;

            const notes = document.getElementById('modal-notes').value;

            fetch(`/api/samples/${currentModalSample}`, {
                method: 'PATCH',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({notes: notes})
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    showToast('Notes saved!', 'success');
                } else {
                    showToast('Failed to save notes: ' + data.error, 'error');
                }
            });
        }

        function deleteFromModal() {
            if (!currentModalSample) return;

            if (!confirm('Delete this sample? This cannot be undone.')) return;

            fetch('/api/dataset/' + currentModalSample, { method: 'DELETE' })
                .then(r => r.json())
                .then(data => {
                    if (data.success) {
                        showToast('Sample deleted', 'success');
                        closeSampleModal();
                        loadDatasetSamples();
                        fetchEvents();
                    } else {
                        showToast('Failed to delete: ' + data.error, 'error');
                    }
                });
        }

        // Close modal on outside click
        document.getElementById('sample-modal').addEventListener('click', function(e) {
            if (e.target === this) {
                closeSampleModal();
            }
        });

        // ========================================
        // Toast Notifications
        // ========================================
        function showToast(message, type = 'info') {
            // Create toast element
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.textContent = message;
            toast.style.cssText = `
                position: fixed;
                bottom: 20px;
                right: 20px;
                padding: 12px 24px;
                border-radius: 8px;
                color: white;
                font-size: 14px;
                z-index: 10000;
                animation: slideIn 0.3s ease;
                background: ${type === 'success' ? '#00ff88' : type === 'error' ? '#ff6b6b' : '#00d4ff'};
                color: ${type === 'success' ? '#000' : type === 'error' ? '#fff' : '#000'};
            `;

            document.body.appendChild(toast);

            // Remove after 3 seconds
            setTimeout(() => {
                toast.style.animation = 'slideOut 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }, 3000);
        }

        // Add animation styles
        const toastStyle = document.createElement('style');
        toastStyle.textContent = `
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOut {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(toastStyle);

        // ========================================
        // Initialization
        // ========================================
        // Fetch dataset on load and periodically
        fetchDataset();
        setInterval(fetchDataset, 5000);

        // Initialize settings slider displays
        document.getElementById('conf-min')?.addEventListener('input', () => updateConfDisplay('min'));
        document.getElementById('conf-max')?.addEventListener('input', () => updateConfDisplay('max'));
    </script>
</body>
</html>
'''


# ============================================
# Unified Detector Wrapper
# ============================================

class UnifiedBeepDetector:
    """
    Unified wrapper that provides a consistent interface for both
    EnsembleBeepDetector and legacy NeuralBeepDetector.

    This allows seamless switching between detection methods via CLI flag.
    """

    def __init__(
        self,
        use_ensemble: bool = True,
        ensemble_threshold: float = 0.35,
        ensemble_yamnet_weight: float = 0.6,
        ensemble_frequency_weight: float = 0.2,
        ensemble_energy_weight: float = 0.2,
        legacy_model_path: str = "models/beep_detector.keras",
        legacy_confidence_threshold: float = 0.5,
        sample_rate: int = 16000,
        window_duration_ms: int = 500,
    ):
        """
        Initialize unified detector.

        Args:
            use_ensemble: If True, use EnsembleBeepDetector; else use legacy NeuralBeepDetector
            ensemble_threshold: Detection threshold for ensemble detector
            ensemble_yamnet_weight: Weight for YAMNet component
            ensemble_frequency_weight: Weight for frequency detector
            ensemble_energy_weight: Weight for energy detector
            legacy_model_path: Path to legacy TFLite/Keras model
            legacy_confidence_threshold: Threshold for legacy detector
            sample_rate: Audio sample rate in Hz
            window_duration_ms: Detection window duration in milliseconds
        """
        self.sample_rate = sample_rate
        self.window_duration_ms = window_duration_ms
        self.use_ensemble = use_ensemble and ENSEMBLE_AVAILABLE
        self.detector = None
        self.detection_count = 0

        # Audio buffer for ensemble detector (processes full audio at once)
        self.audio_buffer = np.array([], dtype=np.int16)
        self.window_samples = int(sample_rate * window_duration_ms / 1000)

        if self.use_ensemble:
            print("\n[DETECTOR] Initializing Ensemble Detector (YAMNet + Frequency + Energy)")
            try:
                # Import ensemble detector only when needed (avoids AVX requirement on legacy CPUs)
                from ensemble_detector import EnsembleBeepDetector

                self.detector = EnsembleBeepDetector(
                    yamnet_weight=ensemble_yamnet_weight,
                    frequency_weight=ensemble_frequency_weight,
                    energy_weight=ensemble_energy_weight,
                    threshold=ensemble_threshold,
                )
                print(f"[DETECTOR] Ensemble detector initialized successfully")
                print(f"[DETECTOR] Threshold: {ensemble_threshold}")
            except Exception as e:
                print(f"[ERROR] Failed to initialize ensemble detector: {e}")
                print(f"[ERROR] Falling back to legacy NeuralBeepDetector")
                self.use_ensemble = False

        if not self.use_ensemble:
            print("\n[DETECTOR] Initializing Legacy NeuralBeepDetector")
            self.detector = NeuralBeepDetector(
                model_path=legacy_model_path,
                sample_rate=sample_rate,
                window_duration_ms=window_duration_ms,
                confidence_threshold=legacy_confidence_threshold,
            )
            print(f"[DETECTOR] Legacy detector initialized")

    @property
    def confidence_threshold(self):
        """Get confidence threshold from wrapped detector."""
        if self.use_ensemble:
            return self.detector.threshold if self.detector else 0.0
        else:
            return self.detector.confidence_threshold if self.detector else 0.0

    def detect(self, samples: np.ndarray) -> dict:
        """
        Run detection on audio samples.

        Args:
            samples: Audio samples (int16 numpy array)

        Returns:
            Dictionary with detection results:
            - detected: bool (is beep detected)
            - confidence: float (0.0-1.0 confidence score)
            - components: dict (ensemble component scores, if using ensemble)
            - buffering: bool (if still buffering audio)
        """
        if self.use_ensemble:
            return self._detect_ensemble(samples)
        else:
            return self.detector.detect(samples)

    def _detect_ensemble(self, samples: np.ndarray) -> dict:
        """Run ensemble detection on audio samples."""
        # Accumulate audio in buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, samples])

        # Need enough audio for meaningful detection
        if len(self.audio_buffer) < self.window_samples:
            return {
                "detected": False,
                "confidence": 0.0,
                "buffering": True,
                "components": {}
            }

        # Use most recent window
        window = self.audio_buffer[-self.window_samples:]

        # Remove DC offset before normalization (critical for detection accuracy!)
        window_centered = window - np.mean(window)
        audio_float = window_centered.astype(np.float32) / 32768.0

        # Run ensemble detection
        try:
            ensemble_score, components = self.detector.detect_beep(
                audio_float,
                self.sample_rate
            )

            is_beep = components['is_beep']

            if is_beep:
                self.detection_count += 1

            # Keep buffer size manageable (2x window)
            max_buffer = self.window_samples * 2
            if len(self.audio_buffer) > max_buffer:
                self.audio_buffer = self.audio_buffer[-max_buffer:]

            return {
                "detected": is_beep,
                "confidence": ensemble_score,
                "components": components,
                "buffering": False,
            }

        except Exception as e:
            print(f"[ERROR] Ensemble detection failed: {e}")
            return {
                "detected": False,
                "confidence": 0.0,
                "error": str(e),
                "buffering": False,
            }

    def reload_model(self, model_path: str = None):
        """Reload model (only for legacy detector)."""
        if not self.use_ensemble and hasattr(self.detector, 'reload_model'):
            self.detector.reload_model(model_path)
        else:
            print("[WARNING] Model reload not supported for ensemble detector")

    def get_config(self) -> dict:
        """Get detector configuration."""
        config = {
            "type": "ensemble" if self.use_ensemble else "legacy",
            "sample_rate": self.sample_rate,
            "window_duration_ms": self.window_duration_ms,
        }

        if self.use_ensemble and hasattr(self.detector, 'get_config'):
            config["ensemble"] = self.detector.get_config()

        return config


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
        use_ensemble: bool = True,
        ensemble_threshold: float = 0.35,
        ensemble_yamnet_weight: float = 0.6,
        ensemble_frequency_weight: float = 0.2,
        ensemble_energy_weight: float = 0.2,
    ):
        self.port = port
        self.response_port = response_port
        self.sample_rate = sample_rate
        self.record_dir = record_dir
        self.web_port = web_port

        # Create unified detector (ensemble or legacy)
        self.nn_detector = UnifiedBeepDetector(
            use_ensemble=use_ensemble,
            ensemble_threshold=ensemble_threshold,
            ensemble_yamnet_weight=ensemble_yamnet_weight,
            ensemble_frequency_weight=ensemble_frequency_weight,
            ensemble_energy_weight=ensemble_energy_weight,
            legacy_model_path=model_path,
            legacy_confidence_threshold=confidence_threshold,
            sample_rate=sample_rate,
            window_duration_ms=window_ms,
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
        self.audio_write_position = 0  # Total samples written (for cursor tracking)

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

        @self.flask_app.route('/api/packet-stats')
        def api_packet_stats():
            """Return detailed packet loss statistics for diagnostics."""
            elapsed = time.time() - self.start_time if self.start_time else 0
            total_expected = self.packets_received + self.packets_lost

            # Calculate rates
            pps = self.packets_received / elapsed if elapsed > 0 else 0
            kbps = (self.bytes_received * 8 / 1000) / elapsed if elapsed > 0 else 0
            loss_pct = (self.packets_lost / total_expected * 100) if total_expected > 0 else 0

            # Expected rate with flow control: batch_count=4, send_interval_ms=50 → ~20 pkt/s
            # (Previously 62.5 pkt/s without batching, caused WiFi buffer exhaustion)
            expected_pps = 20.0
            actual_vs_expected = (pps / expected_pps * 100) if expected_pps > 0 else 0

            # Generate recommendations
            recommendations = []
            if loss_pct > 50:
                recommendations.append("CRITICAL: >50% packet loss. Check WiFi signal, reduce chunk_size, or increase UDP buffer.")
            elif loss_pct > 20:
                recommendations.append("HIGH: 20-50% packet loss. May affect detection accuracy.")
            elif loss_pct > 5:
                recommendations.append("MODERATE: 5-20% packet loss. Some audio gaps expected.")

            if pps < expected_pps * 0.6:
                recommendations.append(f"LOW RATE: Receiving {pps:.1f} pkt/s vs expected {expected_pps:.1f}. Network or ESP32 issue.")

            if not recommendations:
                recommendations.append("OK: Packet loss is within acceptable range.")

            return jsonify({
                "connected": self.esp32_addr is not None,
                "esp32_ip": self.esp32_addr[0] if self.esp32_addr else None,
                "uptime_seconds": round(elapsed, 1),
                "packets": {
                    "received": self.packets_received,
                    "lost": self.packets_lost,
                    "total_expected": total_expected,
                    "loss_percent": round(loss_pct, 2),
                },
                "rates": {
                    "packets_per_sec": round(pps, 2),
                    "expected_pps": expected_pps,
                    "actual_vs_expected_percent": round(actual_vs_expected, 1),
                    "kbps": round(kbps, 2),
                },
                "buffer": {
                    "current_samples": len(self.continuous_buffer),
                    "max_samples": self.continuous_buffer.maxlen,
                    "write_position": self.audio_write_position,
                    "buffer_seconds": round(len(self.continuous_buffer) / self.sample_rate, 2),
                },
                "detection": {
                    "nn_inferences": self.nn_inferences,
                    "nn_rate": round(self.nn_inferences / elapsed, 1) if elapsed > 0 else 0,
                    "detection_count": self.nn_detector.detection_count if hasattr(self.nn_detector, 'detection_count') else 0,
                },
                "recommendations": recommendations
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
            # Capture 5 seconds like auto-detection
            capture_duration = self.labeling_store.settings.get('capture_duration_seconds', 5)
            samples_needed = self.sample_rate * capture_duration

            if len(self.continuous_buffer) < samples_needed:
                return jsonify({
                    "success": False,
                    "error": f"Not enough audio buffered yet (need {capture_duration}s)"
                })

            # Capture audio ending now
            audio_samples = np.array(
                list(self.continuous_buffer)[-samples_needed:],
                dtype=np.int16
            )

            # Create event with confidence=0 to indicate manual capture
            # For manual captures, the "beep" is assumed to be at the end
            event = self.labeling_store.add_event(
                confidence=0.0,  # 0 confidence = manual capture (model missed it)
                audio_samples=audio_samples,
                sample_rate=self.sample_rate,
                source_type="manual_capture",
                detection_offset_ms=int((capture_duration * 1000) - 500),  # Beep at end
            )

            # Auto-label as true positive since user clicked "Mark Beep Now"
            self.labeling_store.label_event(event.id, is_beep=True)

            print(f"\n*** MANUAL BEEP MARKED by user ***")
            print(f"    Saved {capture_duration}s audio: {event.id}")
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

        @self.flask_app.route('/api/audio-stream')
        def api_audio_stream():
            """Return live audio chunk for browser playback with cursor support."""
            import base64

            # ~100ms of audio (1600 samples at 16kHz)
            chunk_size = 1600
            buffer_len = len(self.continuous_buffer)

            if buffer_len < chunk_size:
                return jsonify({'audio': None, 'error': 'Not enough audio buffered'})

            cursor = request.args.get('cursor', type=int)

            if cursor is not None:
                # Cursor-based read: return sequential chunk from cursor position
                # Calculate how far back from the current write position
                samples_behind = self.audio_write_position - cursor

                if samples_behind <= 0:
                    # Client is caught up or ahead, return empty
                    return jsonify({
                        'audio': None,
                        'waiting': True,
                        'next_cursor': cursor
                    })

                if samples_behind > buffer_len:
                    # Client cursor is too old (data was overwritten), reset to recent
                    samples_behind = min(chunk_size * 5, buffer_len)  # Give them some buffer
                    cursor = self.audio_write_position - samples_behind

                # Get the chunk starting from cursor position
                # samples_behind = how far back the cursor is from current write position
                # We want to read chunk_size samples starting from (buffer_end - samples_behind)
                start_offset = buffer_len - samples_behind
                end_offset = min(start_offset + chunk_size, buffer_len)
                actual_chunk = end_offset - start_offset

                buffer_list = list(self.continuous_buffer)
                samples = buffer_list[start_offset:end_offset]
                next_cursor = cursor + actual_chunk
            else:
                # No cursor: legacy behavior - return tail and provide initial cursor
                samples = list(self.continuous_buffer)[-chunk_size:]
                next_cursor = self.audio_write_position

            audio_data = np.array(samples, dtype=np.int16)
            return jsonify({
                'audio': base64.b64encode(audio_data.tobytes()).decode(),
                'sample_rate': self.sample_rate,
                'samples': len(samples),
                'next_cursor': next_cursor,
                'write_position': self.audio_write_position
            })

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

        # ============================================
        # New API Endpoints for Enhanced Dashboard
        # ============================================

        @self.flask_app.route('/api/samples')
        def api_samples():
            """Get filtered and paginated samples."""
            sample_type = request.args.get('type', 'all')
            source = request.args.get('source', 'all')
            conf_min = float(request.args.get('conf_min', 0.0))
            conf_max = float(request.args.get('conf_max', 1.0))
            page = int(request.args.get('page', 1))
            per_page = int(request.args.get('per_page', 20))

            result = self.labeling_store.get_samples(
                sample_type=sample_type,
                source=source,
                conf_min=conf_min,
                conf_max=conf_max,
                page=page,
                per_page=per_page,
            )
            return jsonify(result)

        @self.flask_app.route('/api/samples/<sample_id>/waveform')
        def api_sample_waveform(sample_id):
            """Get waveform data for visualization."""
            num_points = int(request.args.get('points', 500))
            result = self.labeling_store.get_waveform(sample_id, num_points)
            if result is None:
                return jsonify({"error": "Sample not found"}), 404
            return jsonify(result)

        @self.flask_app.route('/api/samples/<sample_id>', methods=['PATCH'])
        def api_update_sample(sample_id):
            """Update a sample's label or notes."""
            data = request.get_json()
            label = data.get('label')  # Can be True, False, or None
            notes = data.get('notes')

            success = self.labeling_store.update_sample(sample_id, label=label, notes=notes)
            if not success:
                return jsonify({"success": False, "error": "Sample not found"}), 404
            return jsonify({"success": True})

        @self.flask_app.route('/api/samples/batch-label', methods=['POST'])
        def api_batch_label():
            """Batch label multiple samples."""
            data = request.get_json()
            sample_ids = data.get('ids', [])
            label = data.get('label')

            if label is None:
                return jsonify({"success": False, "error": "Label required"}), 400

            count = self.labeling_store.batch_label(sample_ids, label)
            return jsonify({"success": True, "updated": count})

        @self.flask_app.route('/api/samples/batch-delete', methods=['POST'])
        def api_batch_delete():
            """Batch delete multiple samples."""
            data = request.get_json()
            sample_ids = data.get('ids', [])

            count = self.labeling_store.batch_delete(sample_ids)
            return jsonify({"success": True, "deleted": count})

        @self.flask_app.route('/api/training-history')
        def api_training_history():
            """Get training run history."""
            return jsonify({
                "history": self.labeling_store.training_history,
                "total_runs": len(self.labeling_store.training_history),
            })

        @self.flask_app.route('/api/settings')
        def api_get_settings():
            """Get current capture settings."""
            return jsonify(self.labeling_store.settings)

        @self.flask_app.route('/api/settings', methods=['POST'])
        def api_update_settings():
            """Update capture settings."""
            data = request.get_json()
            updated = self.labeling_store.update_settings(data)
            return jsonify({"success": True, "settings": updated})

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
        self.audio_write_position += len(samples)  # Track for cursor-based reads

        nn_result = self.nn_detector.detect(samples)
        self.nn_inferences += 1

        if not nn_result.get("buffering", False):
            detected = nn_result["detected"]
            confidence = nn_result["confidence"]
            components = nn_result.get("components", {})
            self.current_confidence = confidence

            self._print_status(sequence, confidence, detected, components)

            if detected != self.last_detection_state or (detected and confidence > 0.9):
                self._send_detection_to_esp32(detected, confidence)
                self.last_detection_state = detected

                if detected:
                    print(f"\n*** BEEP DETECTED! confidence={confidence:.3f} ***")
                    print(f"    Sent to ESP32 at {self.esp32_addr[0]}:{self.response_port}")

                    # Save detection for labeling based on configurable settings
                    # Capture 5 seconds of audio ending at detection moment
                    capture_duration = self.labeling_store.settings.get('capture_duration_seconds', 5)
                    samples_needed = self.sample_rate * capture_duration

                    if self.labeling_store.should_capture(confidence) and len(self.continuous_buffer) >= samples_needed:
                        audio_samples = np.array(
                            list(self.continuous_buffer)[-samples_needed:],
                            dtype=np.int16
                        )
                        # Detection occurs at end of clip (minus detection window ~500ms)
                        detection_offset_ms = int((capture_duration * 1000) - 500)
                        event = self.labeling_store.add_event(
                            confidence=confidence,
                            audio_samples=audio_samples,
                            sample_rate=self.sample_rate,
                            source_type="auto_detection",
                            detection_offset_ms=detection_offset_ms,
                        )
                        print(f"    Saved {capture_duration}s audio for labeling: {event.id}")
                        print(f"    (Detection at {detection_offset_ms}ms mark)\n")
                    else:
                        settings = self.labeling_store.settings
                        if not settings.get('auto_capture_enabled', True):
                            print(f"    (Auto-capture disabled)\n")
                        else:
                            conf_min = settings.get('capture_confidence_min', 0.0)
                            conf_max = settings.get('capture_confidence_max', 1.0)
                            print(f"    (Confidence {confidence:.2f} outside capture range [{conf_min:.2f}-{conf_max:.2f}])\n")

        if self.recording:
            self.record_buffer.append(samples)

    def _print_status(self, seq: int, confidence: float, detected: bool, components: dict = None):
        """Print real-time NN detection status."""
        if detected:
            color = "\033[92m"
        elif confidence > 0.25:  # Reasonable threshold for ensemble
            color = "\033[93m"
        else:
            color = "\033[0m"

        bar_len = min(50, int(confidence * 50))
        bar = "#" * bar_len + "-" * (50 - bar_len)

        status = "BEEP!" if detected else "     "

        # If ensemble detector, show component scores
        if components and 'yamnet' in components:
            yamnet_score = components.get('yamnet', 0.0)
            freq_score = components.get('frequency', 0.0)
            energy_score = components.get('energy', 0.0)
            print(
                f"{color}seq={seq:8d} | ens={confidence:.3f} | Y:{yamnet_score:.2f} F:{freq_score:.2f} E:{energy_score:.2f} | [{bar}] {status}\033[0m",
                end="\r",
            )
        else:
            # Legacy display format
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
        help="Confidence threshold for legacy NN detection (default: 0.7)",
    )
    parser.add_argument(
        "--test-file", type=str, default=None,
        help="Test detection on an audio file instead of live UDP stream",
    )

    # Ensemble detector options
    parser.add_argument(
        "--use-ensemble", action="store_true", default=True,
        help="Use ensemble detector (YAMNet + Frequency + Energy) instead of legacy model (default: True)",
    )
    parser.add_argument(
        "--no-ensemble", action="store_false", dest="use_ensemble",
        help="Use legacy NeuralBeepDetector instead of ensemble detector",
    )
    parser.add_argument(
        "--ensemble-threshold", type=float, default=0.35,
        help="Detection threshold for ensemble detector (default: 0.35)",
    )
    parser.add_argument(
        "--yamnet-weight", type=float, default=0.6,
        help="Weight for YAMNet component in ensemble (default: 0.6)",
    )
    parser.add_argument(
        "--frequency-weight", type=float, default=0.2,
        help="Weight for frequency detector in ensemble (default: 0.2)",
    )
    parser.add_argument(
        "--energy-weight", type=float, default=0.2,
        help="Weight for energy detector in ensemble (default: 0.2)",
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
        use_ensemble=args.use_ensemble,
        ensemble_threshold=args.ensemble_threshold,
        ensemble_yamnet_weight=args.yamnet_weight,
        ensemble_frequency_weight=args.frequency_weight,
        ensemble_energy_weight=args.energy_weight,
    )

    server.start()


if __name__ == "__main__":
    exit(main() or 0)
