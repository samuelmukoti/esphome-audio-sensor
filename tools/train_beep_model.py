#!/usr/bin/env python3
"""
Train Beep Detection Neural Network

This script:
1. Loads training data prepared by prepare_training_data.py
2. Builds a TinyML-compatible CNN model
3. Trains with early stopping
4. Exports as Keras and TFLite (quantized) models

Usage:
    python train_beep_model.py
    python train_beep_model.py --data-dir models/training_data --epochs 100
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks


def build_model(input_shape: tuple[int, int]) -> keras.Model:
    """
    Build a TinyML-compatible CNN for beep detection.

    Memory-optimized architecture for ESP32 (original, not S3):
    - Only 2 Conv1D layers (reduced from 3)
    - Small filter counts (8 channels max)
    - GlobalAveragePooling for minimal parameters
    - Single dense layer before output

    Input shape: (n_frames, n_mfcc) e.g., (25, 20) for 250ms window
    Target memory: <25KB total heap usage
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),

        # First conv block: 8 filters
        layers.Conv1D(8, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(pool_size=2),

        # Second conv block: 8 filters (keep small for memory)
        layers.Conv1D(8, kernel_size=3, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Global pooling (much more efficient than Flatten)
        layers.GlobalAveragePooling1D(),

        # Single dense layer (reduced from 2)
        layers.Dense(8),
        layers.ReLU(),
        layers.Dropout(0.3),

        # Output
        layers.Dense(1, activation='sigmoid')
    ])

    return model


def train_model(model: keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                epochs: int = 100,
                batch_size: int = 32) -> keras.callbacks.History:
    """Train the model with early stopping."""

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    # Class weights (handle imbalance)
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    total = n_positive + n_negative

    class_weight = {
        0: total / (2 * n_negative) if n_negative > 0 else 1.0,
        1: total / (2 * n_positive) if n_positive > 0 else 1.0
    }
    print(f"\nClass weights: {class_weight}")

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history


def evaluate_model(model: keras.Model,
                   X_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model on test set."""
    print("\nEvaluating on test set...")

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss: {results[0]:.4f}")
    print(f"  Accuracy: {results[1]:.4f}")
    print(f"  Precision: {results[2]:.4f}")
    print(f"  Recall: {results[3]:.4f}")

    # Detailed predictions
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    # Confusion matrix
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))

    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {tp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")

    if tp + fn > 0:
        print(f"\n  Detection Rate (Recall): {tp / (tp + fn) * 100:.1f}%")
    if tp + fp > 0:
        print(f"  Precision: {tp / (tp + fp) * 100:.1f}%")


def export_tflite(model: keras.Model,
                  X_train: np.ndarray,
                  output_path: str):
    """Export model as quantized TFLite for ESP32."""
    print(f"\nExporting quantized TFLite model...")

    # Create converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Use representative dataset for full integer quantization
    def representative_dataset():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset

    # Full integer quantization (best for ESP32)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert
    try:
        tflite_model = converter.convert()

        # Save
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"  Saved to: {output_path}")
        print(f"  Size: {size_kb:.1f} KB")

        if size_kb > 100:
            print(f"  WARNING: Model size exceeds 100KB target for ESP32!")
        else:
            print(f"  Model fits ESP32 constraints!")

    except Exception as e:
        print(f"  WARNING: Full INT8 quantization failed: {e}")
        print(f"  Trying float16 quantization instead...")

        # Fallback to float16
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"  Saved to: {output_path}")
        print(f"  Size: {size_kb:.1f} KB (float16)")


def export_tflite_c_array(tflite_path: str, output_path: str):
    """Convert TFLite model to C array for ESP32 embedding."""
    print(f"\nExporting as C array for ESP32...")

    with open(tflite_path, 'rb') as f:
        tflite_model = f.read()

    # Generate C array
    c_array = "// Auto-generated by train_beep_model.py\n"
    c_array += "#pragma once\n\n"
    c_array += f"const unsigned int beep_model_len = {len(tflite_model)};\n"
    c_array += "alignas(8) const unsigned char beep_model[] = {\n"

    # Write bytes in rows of 12
    for i in range(0, len(tflite_model), 12):
        chunk = tflite_model[i:i+12]
        c_array += "    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",\n"

    c_array += "};\n"

    with open(output_path, 'w') as f:
        f.write(c_array)

    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train beep detection model')
    parser.add_argument('--data-dir', default='models/training_data',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', default='models',
                        help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')

    args = parser.parse_args()

    # Load training data
    print("Loading training data...")
    try:
        X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
    except FileNotFoundError:
        print(f"ERROR: Training data not found in {args.data_dir}")
        print("Run prepare_training_data.py first!")
        return 1

    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Val: {X_val.shape}, {y_val.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")

    # Build model
    input_shape = X_train.shape[1:]  # (n_frames, n_mfcc)
    print(f"\nBuilding model with input shape: {input_shape}")
    model = build_model(input_shape)

    # Train
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Save Keras model
    os.makedirs(args.output_dir, exist_ok=True)
    keras_path = os.path.join(args.output_dir, 'beep_detector.keras')
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")

    # Export TFLite
    tflite_path = os.path.join(args.output_dir, 'beep_detector.tflite')
    export_tflite(model, X_train, tflite_path)

    # Export C array for ESP32
    c_array_path = os.path.join(args.output_dir, 'beep_model_data.h')
    export_tflite_c_array(tflite_path, c_array_path)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)
    print(f"\nFiles created:")
    print(f"  - {keras_path} (for Python testing)")
    print(f"  - {tflite_path} (for ESP32 deployment)")
    print(f"  - {c_array_path} (C header for ESP32)")
    print(f"\nNext step: Test with audio_server.py --use-nn")

    return 0


if __name__ == '__main__':
    exit(main())
