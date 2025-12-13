#!/usr/bin/env python3
"""
Export Keras model weights to C header for manual ESP32 inference.

This script extracts weights from the trained Keras model and generates
a C header file with float arrays for each layer.

Usage:
    python export_model_weights.py models_250ms/beep_detector.keras
"""

import argparse
import os
import numpy as np
from tensorflow import keras


def export_weights_to_c_header(model: keras.Model, output_path: str):
    """Export all model weights to a C header file."""

    lines = [
        "// Auto-generated model weights for beep_detector_nn",
        "// Model: 2-layer CNN for 250ms window (25 frames x 20 MFCC)",
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
    ]

    # Track layer names for the header
    weight_info = []

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        layer_name = layer.name.replace('/', '_')
        print(f"Processing layer: {layer.name} ({layer.__class__.__name__})")

        if isinstance(layer, keras.layers.Conv1D):
            # Conv1D weights: (kernel_size, in_channels, out_channels)
            kernel, bias = weights
            print(f"  Kernel shape: {kernel.shape}, Bias shape: {bias.shape}")

            # Flatten kernel for C array
            kernel_flat = kernel.flatten()
            lines.append(f"// {layer_name}: kernel shape {kernel.shape}")
            lines.append(f"static const float {layer_name}_kernel[] = {{")
            lines.append(format_array(kernel_flat))
            lines.append("};")
            lines.append(f"static const float {layer_name}_bias[] = {{")
            lines.append(format_array(bias))
            lines.append("};")
            lines.append("")

            weight_info.append((f"{layer_name}_kernel", kernel_flat.size))
            weight_info.append((f"{layer_name}_bias", bias.size))

        elif isinstance(layer, keras.layers.BatchNormalization):
            # BatchNorm weights: gamma, beta, moving_mean, moving_variance
            gamma, beta, mean, var = weights
            print(f"  Gamma: {gamma.shape}, Beta: {beta.shape}")

            lines.append(f"// {layer_name}: {gamma.shape[0]} channels")
            lines.append(f"static const float {layer_name}_gamma[] = {{")
            lines.append(format_array(gamma))
            lines.append("};")
            lines.append(f"static const float {layer_name}_beta[] = {{")
            lines.append(format_array(beta))
            lines.append("};")
            lines.append(f"static const float {layer_name}_mean[] = {{")
            lines.append(format_array(mean))
            lines.append("};")
            lines.append(f"static const float {layer_name}_var[] = {{")
            lines.append(format_array(var))
            lines.append("};")
            lines.append("")

            weight_info.append((f"{layer_name}_gamma", gamma.size))
            weight_info.append((f"{layer_name}_beta", beta.size))
            weight_info.append((f"{layer_name}_mean", mean.size))
            weight_info.append((f"{layer_name}_var", var.size))

        elif isinstance(layer, keras.layers.Dense):
            # Dense weights: (in_features, out_features), bias
            kernel, bias = weights
            print(f"  Kernel shape: {kernel.shape}, Bias shape: {bias.shape}")

            kernel_flat = kernel.flatten()
            lines.append(f"// {layer_name}: {kernel.shape[0]} -> {kernel.shape[1]}")
            lines.append(f"static const float {layer_name}_kernel[] = {{")
            lines.append(format_array(kernel_flat))
            lines.append("};")
            lines.append(f"static const float {layer_name}_bias[] = {{")
            lines.append(format_array(bias))
            lines.append("};")
            lines.append("")

            weight_info.append((f"{layer_name}_kernel", kernel_flat.size))
            weight_info.append((f"{layer_name}_bias", bias.size))

    # Add summary
    total_params = sum(size for _, size in weight_info)
    total_bytes = total_params * 4  # 4 bytes per float

    lines.insert(5, f"// Total parameters: {total_params}")
    lines.insert(6, f"// Total size: {total_bytes} bytes ({total_bytes/1024:.1f} KB)")
    lines.insert(7, "")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\nExported weights to: {output_path}")
    print(f"Total parameters: {total_params}")
    print(f"Total size: {total_bytes} bytes ({total_bytes/1024:.1f} KB)")

    return weight_info


def format_array(arr: np.ndarray, per_line: int = 8) -> str:
    """Format numpy array as comma-separated C values."""
    flat = arr.flatten()
    lines = []
    for i in range(0, len(flat), per_line):
        chunk = flat[i:i+per_line]
        formatted = ", ".join(f"{v:.8f}f" for v in chunk)
        lines.append(f"    {formatted},")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Export Keras model weights to C header')
    parser.add_argument('model_path', help='Path to Keras model file')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path for C header (default: model_weights.h in same dir)')

    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(args.model_path)

    print("\nModel summary:")
    model.summary()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        model_dir = os.path.dirname(args.model_path)
        output_path = os.path.join(model_dir, 'model_weights.h')

    # Export weights
    export_weights_to_c_header(model, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
