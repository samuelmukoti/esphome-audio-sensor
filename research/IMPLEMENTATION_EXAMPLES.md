# Implementation Examples: Beep Detection on M5Stack Atom Echo

## Example 1: Basic ESPHome Configuration

### Complete YAML Configuration

```yaml
substitutions:
  name: atom-echo-beep
  friendly_name: Appliance Beep Detector

esphome:
  name: ${name}
  friendly_name: ${friendly_name}
  platformio_options:
    board_build.flash_mode: dio
    # Add external libraries if using FFT approach
    lib_deps:
      - https://github.com/MichielFromNL/SoundAnalyzer

esp32:
  board: m5stack-atom
  framework:
    type: esp-idf  # Required for audio features
    version: recommended

# Disable Bluetooth to save resources
esp32_ble_tracker:
  active: false

# Basic connectivity
wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
  ap:
    ssid: "${name} Fallback"
    password: !secret ap_password

api:
  encryption:
    key: !secret api_encryption_key

ota:
  platform: esphome
  password: !secret ota_password

logger:
  level: INFO

# Status LED
light:
  - platform: esp32_rmt_led_strip
    id: led
    name: "Status LED"
    pin: GPIO27
    num_leds: 1
    rmt_channel: 0
    rgb_order: GRB
    chipset: SK6812

# I2S Audio Configuration
i2s_audio:
  i2s_lrclk_pin: GPIO33
  i2s_bclk_pin: GPIO19

# PDM Microphone
microphone:
  - platform: i2s_audio
    id: atom_mic
    i2s_din_pin: GPIO23
    adc_type: external
    pdm: true  # CRITICAL for SPM1423
    sample_rate: 16000
    bits_per_sample: 32bit
    channel: right

# Binary sensor for beep detection
binary_sensor:
  - platform: template
    name: "Appliance Beep Detected"
    id: beep_detected
    device_class: sound

# Sensor for audio level (debugging)
sensor:
  - platform: template
    name: "Audio Level"
    id: audio_level
    unit_of_measurement: "dB"
    accuracy_decimals: 1

  - platform: template
    name: "Beep Confidence"
    id: beep_confidence
    unit_of_measurement: "%"
    accuracy_decimals: 0
```

---

## Example 2: FFT-Based Beep Detection (C++ External Component)

### Directory Structure

```
esphome/
├── atom-echo-beep.yaml
├── custom_components/
│   └── beep_detector/
│       ├── __init__.py
│       ├── beep_detector.h
│       └── beep_detector.cpp
```

### custom_components/beep_detector/__init__.py

```python
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, sensor, microphone
from esphome.const import (
    CONF_ID,
    CONF_MICROPHONE,
    UNIT_DECIBEL,
    UNIT_PERCENT,
)

DEPENDENCIES = ["microphone"]

beep_detector_ns = cg.esphome_ns.namespace("beep_detector")
BeepDetector = beep_detector_ns.class_("BeepDetector", cg.Component)

CONF_BEEP_SENSOR = "beep_sensor"
CONF_CONFIDENCE_SENSOR = "confidence_sensor"
CONF_MIN_FREQUENCY = "min_frequency"
CONF_MAX_FREQUENCY = "max_frequency"
CONF_THRESHOLD = "threshold"
CONF_SAMPLE_WINDOW = "sample_window"

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(BeepDetector),
    cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
    cv.Optional(CONF_BEEP_SENSOR): binary_sensor.binary_sensor_schema(),
    cv.Optional(CONF_CONFIDENCE_SENSOR): sensor.sensor_schema(
        unit_of_measurement=UNIT_PERCENT,
        accuracy_decimals=0,
    ),
    cv.Optional(CONF_MIN_FREQUENCY, default=2000): cv.int_range(min=100, max=8000),
    cv.Optional(CONF_MAX_FREQUENCY, default=4000): cv.int_range(min=100, max=8000),
    cv.Optional(CONF_THRESHOLD, default=0.6): cv.float_range(min=0.1, max=1.0),
    cv.Optional(CONF_SAMPLE_WINDOW, default=1024): cv.one_of(512, 1024, 2048),
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    if CONF_BEEP_SENSOR in config:
        sens = await binary_sensor.new_binary_sensor(config[CONF_BEEP_SENSOR])
        cg.add(var.set_beep_sensor(sens))

    if CONF_CONFIDENCE_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_CONFIDENCE_SENSOR])
        cg.add(var.set_confidence_sensor(sens))

    cg.add(var.set_frequency_range(
        config[CONF_MIN_FREQUENCY],
        config[CONF_MAX_FREQUENCY]
    ))
    cg.add(var.set_threshold(config[CONF_THRESHOLD]))
    cg.add(var.set_sample_window(config[CONF_SAMPLE_WINDOW]))
```

### custom_components/beep_detector/beep_detector.h

```cpp
#pragma once

#include "esphome/core/component.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/microphone/microphone.h"
#include <vector>

namespace esphome {
namespace beep_detector {

class BeepDetector : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::DATA; }

  void set_microphone(microphone::Microphone *mic) { mic_ = mic; }
  void set_beep_sensor(binary_sensor::BinarySensor *sensor) { beep_sensor_ = sensor; }
  void set_confidence_sensor(sensor::Sensor *sensor) { confidence_sensor_ = sensor; }
  void set_frequency_range(uint16_t min_freq, uint16_t max_freq);
  void set_threshold(float threshold) { threshold_ = threshold; }
  void set_sample_window(uint16_t window) { sample_window_ = window; }

 protected:
  microphone::Microphone *mic_{nullptr};
  binary_sensor::BinarySensor *beep_sensor_{nullptr};
  sensor::Sensor *confidence_sensor_{nullptr};

  uint16_t min_frequency_{2000};
  uint16_t max_frequency_{4000};
  float threshold_{0.6};
  uint16_t sample_window_{1024};

  // FFT processing
  std::vector<float> audio_buffer_;
  std::vector<float> fft_output_;

  uint16_t min_bin_{0};
  uint16_t max_bin_{0};

  bool last_state_{false};
  uint32_t last_detection_time_{0};
  static const uint32_t DEBOUNCE_MS = 500;  // 500ms debounce

  void process_audio();
  float compute_fft();
  float calculate_target_energy();
  void update_state(bool detected, float confidence);
};

}  // namespace beep_detector
}  // namespace esphome
```

### custom_components/beep_detector/beep_detector.cpp

```cpp
#include "beep_detector.h"
#include "esphome/core/log.h"
#include <cmath>

// Include FFT library (SoundAnalyzer or ESP-DSP)
// For this example, we'll use pseudo-FFT
// In real implementation, use: #include "SoundAnalyzer.h"

namespace esphome {
namespace beep_detector {

static const char *TAG = "beep_detector";

void BeepDetector::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Beep Detector...");

  // Allocate buffers
  audio_buffer_.resize(sample_window_);
  fft_output_.resize(sample_window_ / 2);

  // Calculate FFT bin indices for target frequency range
  float sample_rate = 16000.0f;  // From microphone config
  float bin_resolution = sample_rate / sample_window_;

  min_bin_ = static_cast<uint16_t>(min_frequency_ / bin_resolution);
  max_bin_ = static_cast<uint16_t>(max_frequency_ / bin_resolution);

  ESP_LOGCONFIG(TAG, "  Sample Window: %d", sample_window_);
  ESP_LOGCONFIG(TAG, "  Frequency Range: %d - %d Hz", min_frequency_, max_frequency_);
  ESP_LOGCONFIG(TAG, "  FFT Bins: %d - %d", min_bin_, max_bin_);
  ESP_LOGCONFIG(TAG, "  Threshold: %.2f", threshold_);

  ESP_LOGCONFIG(TAG, "Beep Detector setup complete");
}

void BeepDetector::loop() {
  // Check if microphone has new audio data
  if (mic_ == nullptr || !mic_->is_running()) {
    return;
  }

  // Read audio samples
  // Note: In real implementation, use mic_->read() or similar
  // This is pseudo-code showing the processing flow

  // For demonstration, assume we have audio data in audio_buffer_
  // In real implementation:
  // size_t bytes_read = mic_->read(audio_buffer_.data(), sample_window_ * sizeof(int16_t));

  process_audio();
}

void BeepDetector::set_frequency_range(uint16_t min_freq, uint16_t max_freq) {
  min_frequency_ = min_freq;
  max_frequency_ = max_freq;
}

void BeepDetector::process_audio() {
  // Compute FFT
  float confidence = compute_fft();

  // Determine if beep detected
  bool detected = (confidence > threshold_);

  // Debounce
  uint32_t now = millis();
  if (detected != last_state_) {
    if (now - last_detection_time_ > DEBOUNCE_MS) {
      update_state(detected, confidence);
      last_state_ = detected;
      last_detection_time_ = now;
    }
  }
}

float BeepDetector::compute_fft() {
  // Real implementation would use SoundAnalyzer or ESP-DSP
  // Example with SoundAnalyzer:
  //
  // SoundAnalyzer analyzer(sample_window_);
  // analyzer.computeFFT(audio_buffer_.data());
  // float *fft_magnitudes = analyzer.getFFTMagnitudes();
  //
  // For now, pseudo-code:

  // Calculate total energy
  float total_energy = 0.0f;
  for (size_t i = 0; i < fft_output_.size(); i++) {
    total_energy += fft_output_[i];
  }

  // Calculate target frequency band energy
  float target_energy = calculate_target_energy();

  // Calculate ratio (confidence)
  float ratio = (total_energy > 0) ? (target_energy / total_energy) : 0.0f;

  return ratio;
}

float BeepDetector::calculate_target_energy() {
  float energy = 0.0f;
  for (uint16_t i = min_bin_; i <= max_bin_ && i < fft_output_.size(); i++) {
    energy += fft_output_[i];
  }
  return energy;
}

void BeepDetector::update_state(bool detected, float confidence) {
  ESP_LOGD(TAG, "Beep %s (confidence: %.2f%%)",
           detected ? "DETECTED" : "ended",
           confidence * 100.0f);

  // Update binary sensor
  if (beep_sensor_ != nullptr) {
    beep_sensor_->publish_state(detected);
  }

  // Update confidence sensor
  if (confidence_sensor_ != nullptr) {
    confidence_sensor_->publish_state(confidence * 100.0f);
  }
}

}  // namespace beep_detector
}  // namespace esphome
```

### Usage in YAML

```yaml
external_components:
  - source:
      type: local
      path: custom_components

beep_detector:
  id: my_beep_detector
  microphone: atom_mic
  min_frequency: 2000  # Hz
  max_frequency: 4000  # Hz
  threshold: 0.6       # 60% of energy in target band
  sample_window: 1024  # samples
  beep_sensor:
    name: "Appliance Beep Detected"
  confidence_sensor:
    name: "Beep Detection Confidence"
```

---

## Example 3: Edge Impulse Integration

### YAML Configuration

```yaml
substitutions:
  name: atom-echo-beep-ml
  friendly_name: ML Beep Detector

esphome:
  name: ${name}
  friendly_name: ${friendly_name}
  platformio_options:
    board_build.flash_mode: dio
    lib_deps:
      # Edge Impulse Arduino library (after exporting from Edge Impulse)
      - https://github.com/your-username/ei-appliance-beep-arduino.git

esp32:
  board: m5stack-atom
  framework:
    type: esp-idf

# ... (same I2S and microphone config as Example 1) ...

# Custom component for Edge Impulse inference
external_components:
  - source:
      type: local
      path: custom_components

ei_inference:
  id: my_ei_inference
  microphone: atom_mic
  model: "appliance-beep"  # From Edge Impulse project
  classifications:
    - label: "beep"
      threshold: 0.8
      binary_sensor:
        name: "Beep Detected"
    - label: "background"
      threshold: 0.8
      binary_sensor:
        name: "Background Noise"
```

### Simplified Edge Impulse Wrapper (Pseudo-code)

```cpp
// custom_components/ei_inference/ei_inference.cpp

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"
#include "model-parameters/model_metadata.h"

void EIInference::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Edge Impulse Inference...");

  // Initialize Edge Impulse runtime
  if (ei_classifier_init() != EI_IMPULSE_OK) {
    ESP_LOGE(TAG, "Failed to initialize Edge Impulse classifier");
    return;
  }

  ESP_LOGCONFIG(TAG, "Model: %s", EI_CLASSIFIER_PROJECT_NAME);
  ESP_LOGCONFIG(TAG, "Input size: %d", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
  ESP_LOGCONFIG(TAG, "Classes: %d", EI_CLASSIFIER_LABEL_COUNT);
}

void EIInference::process_audio() {
  // Prepare input features (MFE/MFCC from audio buffer)
  signal_t signal;
  signal.total_length = audio_buffer_.size();
  signal.get_data = &get_audio_signal_data;

  // Run inference
  ei_impulse_result_t result;
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

  if (res != EI_IMPULSE_OK) {
    ESP_LOGE(TAG, "Inference failed: %d", res);
    return;
  }

  // Process results
  for (size_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    float confidence = result.classification[i].value;
    const char *label = result.classification[i].label;

    ESP_LOGD(TAG, "  %s: %.2f%%", label, confidence * 100.0f);

    // Publish to Home Assistant
    update_classification(label, confidence);
  }
}
```

---

## Example 4: Testing & Debugging Utilities

### Audio Spectrum Analyzer (Debugging Tool)

```yaml
# Add to your YAML for debugging
sensor:
  - platform: template
    name: "FFT Bin 0-500Hz"
    id: fft_bin_0
    unit_of_measurement: "dB"

  - platform: template
    name: "FFT Bin 500-1000Hz"
    id: fft_bin_1
    unit_of_measurement: "dB"

  - platform: template
    name: "FFT Bin 1-2kHz"
    id: fft_bin_2
    unit_of_measurement: "dB"

  - platform: template
    name: "FFT Bin 2-4kHz (Target)"
    id: fft_bin_3
    unit_of_measurement: "dB"

  - platform: template
    name: "FFT Bin 4-8kHz"
    id: fft_bin_4
    unit_of_measurement: "dB"

# Publish FFT bins from C++ component for debugging
# Then visualize in Home Assistant / Grafana
```

### C++ Debugging Helper

```cpp
void BeepDetector::debug_spectrum() {
  // Calculate energy in frequency bands for debugging
  const int num_bands = 5;
  float bands[num_bands] = {0};

  // Band 0: 0-500 Hz
  for (int i = 0; i < 16; i++) bands[0] += fft_output_[i];

  // Band 1: 500-1000 Hz
  for (int i = 16; i < 32; i++) bands[1] += fft_output_[i];

  // Band 2: 1-2 kHz
  for (int i = 32; i < 64; i++) bands[2] += fft_output_[i];

  // Band 3: 2-4 kHz (TARGET)
  for (int i = 64; i < 128; i++) bands[3] += fft_output_[i];

  // Band 4: 4-8 kHz
  for (int i = 128; i < 256; i++) bands[4] += fft_output_[i];

  // Convert to dB and publish
  for (int i = 0; i < num_bands; i++) {
    float db = 20 * log10(bands[i] + 1e-10);
    // Publish to corresponding sensor
  }
}
```

---

## Example 5: Home Assistant Automation

### Binary Sensor Integration

```yaml
# In Home Assistant configuration.yaml

automation:
  - alias: "Notify on Appliance Beep"
    trigger:
      - platform: state
        entity_id: binary_sensor.appliance_beep_detected
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          title: "Appliance Alert"
          message: "Your appliance is beeping!"

  - alias: "Flash LED on Beep"
    trigger:
      - platform: state
        entity_id: binary_sensor.appliance_beep_detected
        to: "on"
    action:
      - service: light.turn_on
        target:
          entity_id: light.atom_echo_status_led
        data:
          rgb_color: [255, 0, 0]
          brightness: 255
      - delay: "00:00:02"
      - service: light.turn_off
        target:
          entity_id: light.atom_echo_status_led
```

### Dashboard Card

```yaml
# Lovelace dashboard card
type: entities
title: Appliance Beep Monitor
entities:
  - entity: binary_sensor.appliance_beep_detected
    name: "Beep Status"
  - entity: sensor.beep_detection_confidence
    name: "Detection Confidence"
  - entity: sensor.audio_level
    name: "Audio Level"
  - type: section
    label: "Debug - Frequency Bands"
  - entity: sensor.fft_bin_2_4khz_target
    name: "Target Band (2-4kHz)"
  - entity: sensor.fft_bin_0_500hz
    name: "Low Freq"
  - entity: sensor.fft_bin_4_8khz
    name: "High Freq"
```

---

## Performance Optimization Tips

### 1. CPU Frequency Scaling

```cpp
// In setup()
setCpuFrequencyMhz(160);  // Reduce from 240 MHz to save power
// Test if inference still completes in time
```

### 2. Buffer Management

```cpp
// Use static buffers (allocated once)
static float audio_buffer[1024];
static float fft_output[512];

// Avoid dynamic allocation in loop()
// BAD:  std::vector<float> temp(1024);
// GOOD: Use pre-allocated buffers
```

### 3. Dual-Core Utilization

```cpp
// Run audio capture on Core 0, processing on Core 1
void setup() {
  xTaskCreatePinnedToCore(
    audio_task,      // Function
    "AudioTask",     // Name
    4096,            // Stack size
    NULL,            // Parameters
    1,               // Priority
    NULL,            // Handle
    0                // Core 0
  );

  xTaskCreatePinnedToCore(
    inference_task,  // Function
    "InferenceTask", // Name
    8192,            // Stack size
    NULL,            // Parameters
    1,               // Priority
    NULL,            // Handle
    1                // Core 1
  );
}
```

### 4. Memory Profiling

```cpp
void BeepDetector::loop() {
  // Periodic memory check
  static uint32_t last_check = 0;
  if (millis() - last_check > 10000) {  // Every 10 seconds
    ESP_LOGD(TAG, "Free heap: %d bytes", ESP.getFreeHeap());
    ESP_LOGD(TAG, "Min free heap: %d bytes", ESP.getMinFreeHeap());
    last_check = millis();
  }

  // ... rest of processing ...
}
```

---

## Troubleshooting Guide

### Issue: No audio captured

```cpp
// Check I2S configuration
ESP_LOGCONFIG(TAG, "I2S Config:");
ESP_LOGCONFIG(TAG, "  LRCK: GPIO%d", 33);
ESP_LOGCONFIG(TAG, "  BCLK: GPIO%d", 19);
ESP_LOGCONFIG(TAG, "  DIN: GPIO%d", 23);
ESP_LOGCONFIG(TAG, "  PDM: %s", pdm_enabled ? "YES" : "NO");

// Verify microphone is reading
int16_t sample;
mic_->read(&sample, sizeof(sample));
ESP_LOGD(TAG, "Sample: %d", sample);
```

### Issue: High false positive rate

```cpp
// Increase threshold
threshold_ = 0.75;  // Was 0.6

// Add hysteresis
static int detection_count = 0;
if (ratio > threshold_) {
  detection_count++;
  if (detection_count > 3) {  // Require 3 consecutive detections
    publish_state(true);
  }
} else {
  detection_count = 0;
  publish_state(false);
}
```

### Issue: Memory overflow

```cpp
// Reduce buffer sizes
sample_window_ = 512;  // Was 1024

// Disable debug logging
logger:
  level: WARN  # Was INFO or DEBUG

// Check for leaks
ESP_LOGW(TAG, "Heap usage increased by %d bytes",
         initial_heap - ESP.getFreeHeap());
```

---

## Next Steps

1. **Start with Example 1**: Basic ESPHome configuration
2. **Test audio capture**: Use voice_assistant to verify microphone works
3. **Implement Example 2**: FFT-based detection for quick prototyping
4. **Collect data**: Record beep samples if moving to ML approach
5. **Train Edge Impulse model**: If needed (Example 3)
6. **Optimize**: Use performance tips and debugging tools (Example 4-5)

---

**Document Status**: Implementation-Ready
**Last Updated**: 2025-12-10
**Framework**: ESPHome + ESP-IDF
