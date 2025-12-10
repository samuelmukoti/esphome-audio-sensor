# ESPHome Beep Detection Architecture
## M5Stack Atom Echo - Water Heater Beep Sensor

**Version:** 1.0
**Target Hardware:** M5Stack Atom Echo (ESP32 + I2S Microphone)
**Framework:** ESPHome with ESP-IDF
**Last Updated:** 2025-12-10

---

## Executive Summary

This document defines the complete architecture for a real-time beep detection system running on M5Stack Atom Echo. The system captures audio via I2S microphone, processes it on-device using efficient signal processing, and exposes a binary sensor to Home Assistant indicating beep presence.

**Key Design Principles:**
- Real-time processing with <500ms latency
- Low memory footprint (<100KB for audio buffers)
- Configurable via YAML without firmware changes
- Reliable detection with minimal false positives
- Graceful degradation under processing constraints

---

## 1. System Architecture Overview

### 1.1 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    Home Assistant                           │
│                  (Binary Sensor Consumer)                   │
└────────────────────────┬────────────────────────────────────┘
                         │ ESPHome API
                         │
┌────────────────────────▼────────────────────────────────────┐
│                  ESPHome Core Layer                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         Binary Sensor Component                      │  │
│  │  (water_heater_beep - ON/OFF state)                  │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │    Custom Beep Detection Component                   │  │
│  │    beep_detector (C++ or Lambda)                     │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │ 1. Audio Capture (I2S Buffer Management)       │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 2. Preprocessing (DC removal, filtering)       │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 3. Feature Extraction (FFT/Goertzel/Energy)    │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 4. Detection Logic (Threshold + Pattern)       │  │  │
│  │  ├────────────────────────────────────────────────┤  │  │
│  │  │ 5. State Management (Debouncing)               │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └───────────────────────┬──────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────▼──────────────────────────────┐  │
│  │          I2S Microphone Driver                       │  │
│  │  (ESP32 I2S peripheral - SPM1423 microphone)         │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
               [Physical Audio Input]
```

### 1.2 Data Flow Pipeline

```
Audio Input (Analog)
    ↓
I2S Microphone (16kHz mono, 16-bit)
    ↓
Circular Buffer (512 samples, 32ms chunks)
    ↓
DC Removal + High-pass Filter (>100Hz)
    ↓
Window Function (Hann/Hamming)
    ↓
Feature Extraction:
  ├─ Option A: Goertzel Algorithm (target frequencies)
  ├─ Option B: FFT (frequency spectrum)
  └─ Option C: RMS Energy (simple threshold)
    ↓
Detection Logic:
  - Frequency match (±50Hz tolerance)
  - Amplitude threshold (dynamic/fixed)
  - Pattern recognition (beep cadence)
    ↓
Debouncing (3-5 consecutive detections)
    ↓
Binary Sensor State Update
    ↓
Home Assistant (via ESPHome API)
```

---

## 2. Hardware Abstraction Layer

### 2.1 M5Stack Atom Echo Specifications

**Microcontroller:**
- ESP32-PICO (240MHz dual-core, 520KB SRAM, 4MB flash)
- I2S peripheral for audio capture
- Wi-Fi for Home Assistant connectivity

**Microphone:**
- SPM1423 I2S MEMS microphone
- Omnidirectional pattern
- Frequency response: 100Hz - 10kHz
- SNR: 61dB typical
- Sensitivity: -26dBFS

**I2S Configuration:**
```yaml
# Hardware constraints from M5Stack Atom Echo
I2S_PINS:
  BCLK: GPIO 19
  LRCLK: GPIO 33
  DATA_IN: GPIO 22
```

### 2.2 Audio Capture Configuration

**Sampling Strategy:**
```
Target Sample Rate: 16 kHz (sufficient for <8kHz beeps)
Bit Depth: 16-bit signed integer
Channels: Mono (I2S left channel)
Buffer Size: 512 samples (32ms @ 16kHz)
DMA Buffers: 4 (double buffering for real-time)
```

**Rationale:**
- 16kHz Nyquist → captures frequencies up to 8kHz
- Lower sample rate reduces CPU/memory vs. 44.1kHz
- 32ms chunks provide good time resolution
- DMA buffers prevent audio dropout during processing

---

## 3. Processing Pipeline Design

### 3.1 Stage 1: Audio Capture

**Implementation:** ESPHome I2S microphone component + custom read handler

```cpp
// Pseudo-code for capture loop
class BeepDetector : public Component {
  static const size_t BUFFER_SIZE = 512;
  int16_t audio_buffer[BUFFER_SIZE];

  void loop() override {
    size_t bytes_read = i2s_read(audio_buffer, BUFFER_SIZE * 2, 10);
    if (bytes_read > 0) {
      process_audio(audio_buffer, BUFFER_SIZE);
    }
  }
};
```

**Memory Budget:** 1KB for audio buffer + 512B for processed samples

### 3.2 Stage 2: Preprocessing

**DC Offset Removal:**
```cpp
// Remove DC bias from microphone
float dc_estimate = 0;
const float DC_ALPHA = 0.995;

for (int i = 0; i < BUFFER_SIZE; i++) {
  dc_estimate = DC_ALPHA * dc_estimate + (1 - DC_ALPHA) * audio_buffer[i];
  audio_buffer[i] -= (int16_t)dc_estimate;
}
```

**High-pass Filter:**
- 1st order IIR filter @ 100Hz cutoff
- Removes low-frequency noise (HVAC, rumble)
- Minimal CPU overhead (~2 operations per sample)

### 3.3 Stage 3: Feature Extraction

**Three Implementation Options:**

#### Option A: Goertzel Algorithm (Recommended)
**Best for:** Known beep frequencies (e.g., 1kHz, 2kHz)

```cpp
// Efficient single-frequency DFT
class GoertzelDetector {
  float coeff;  // Pre-computed: 2 * cos(2π * target_freq / sample_rate)
  float q1, q2;

  float detect(int16_t* samples, size_t count, float target_freq) {
    coeff = 2.0 * cos(2 * PI * target_freq / 16000.0);
    q1 = q2 = 0;

    for (size_t i = 0; i < count; i++) {
      float q0 = coeff * q1 - q2 + samples[i];
      q2 = q1;
      q1 = q0;
    }

    // Return magnitude
    return sqrt(q1*q1 + q2*q2 - q1*q2*coeff);
  }
};
```

**Performance:** ~1500 operations for 512 samples → <0.5ms @ 240MHz
**Memory:** 16 bytes state + stack
**Accuracy:** ±10Hz frequency resolution @ 16kHz/512 samples

#### Option B: FFT (Real-valued, 512-point)
**Best for:** Unknown frequencies or broadband beeps

```cpp
// Using ESP-DSP or similar optimized FFT
#include "esp_dsp.h"

float fft_input[512];
float fft_output[512];

void analyze_spectrum(int16_t* audio, size_t count) {
  // Window function (Hann)
  for (int i = 0; i < count; i++) {
    float w = 0.5 * (1 - cos(2 * PI * i / count));
    fft_input[i] = audio[i] * w;
  }

  // FFT
  dsps_fft2r_fc32(fft_input, count);
  dsps_bit_rev_fc32(fft_input, count);

  // Power spectrum
  for (int i = 0; i < count/2; i++) {
    float re = fft_input[i*2];
    float im = fft_input[i*2+1];
    fft_output[i] = sqrt(re*re + im*im);
  }
}
```

**Performance:** ~15ms for 512-point FFT @ 240MHz (using ESP-DSP)
**Memory:** 4KB for FFT buffers
**Accuracy:** 31.25Hz frequency bins (16000/512)

#### Option C: RMS Energy Detection (Simplest)
**Best for:** Any loud beep regardless of frequency

```cpp
float compute_rms(int16_t* samples, size_t count) {
  float sum_squares = 0;
  for (size_t i = 0; i < count; i++) {
    sum_squares += samples[i] * samples[i];
  }
  return sqrt(sum_squares / count);
}
```

**Performance:** ~0.1ms for 512 samples
**Memory:** 4 bytes
**Accuracy:** No frequency selectivity

**Recommendation Hierarchy:**
1. **Start with Option C (RMS)** - simplest, validate basic detection
2. **Upgrade to Option A (Goertzel)** - if specific frequency known
3. **Use Option B (FFT)** - if need full spectrum or unknown frequency

### 3.4 Stage 4: Detection Logic

**Multi-criteria Detection:**

```cpp
class BeepDetector {
  // Configurable thresholds
  float energy_threshold;      // RMS amplitude
  float freq_min, freq_max;    // Frequency range (Hz)
  uint8_t min_detections;      // Consecutive hits needed
  uint16_t debounce_ms;        // Time filtering

  // State
  uint8_t consecutive_hits;
  uint32_t last_detect_time;

  bool check_beep(float energy, float frequency) {
    bool detected = false;

    // Criteria 1: Energy above threshold
    if (energy > energy_threshold) {

      // Criteria 2: Frequency in range (if using Goertzel/FFT)
      if (frequency >= freq_min && frequency <= freq_max) {
        consecutive_hits++;

        // Criteria 3: Sustained detection
        if (consecutive_hits >= min_detections) {
          detected = true;
        }
      } else {
        consecutive_hits = 0;
      }
    } else {
      consecutive_hits = max(0, consecutive_hits - 1); // Gradual decay
    }

    // Debounce
    uint32_t now = millis();
    if (detected && (now - last_detect_time) > debounce_ms) {
      last_detect_time = now;
      return true;
    }

    return false;
  }
};
```

**Pattern Recognition (Advanced):**
For repeating beep patterns (e.g., 3 beeps, 2s pause, repeat):

```cpp
class BeepPatternMatcher {
  struct BeepEvent {
    uint32_t timestamp;
    uint16_t duration_ms;
  };

  std::vector<BeepEvent> recent_beeps;

  bool matches_pattern(uint16_t expected_count,
                       uint16_t interval_ms,
                       uint16_t tolerance_ms) {
    // Check if recent beeps match expected temporal pattern
    // Returns true if pattern detected
  }
};
```

### 3.5 Stage 5: State Management

**Binary Sensor States:**
- `ON` - Beep currently detected
- `OFF` - No beep detected

**Hysteresis Logic:**
```cpp
// Prevent rapid ON/OFF toggling
class StateManager {
  bool current_state = false;
  uint32_t state_change_time = 0;

  const uint16_t MIN_ON_TIME = 100;   // ms - minimum beep duration
  const uint16_t MIN_OFF_TIME = 500;  // ms - minimum silence

  bool update_state(bool new_detection) {
    uint32_t now = millis();

    if (new_detection && !current_state) {
      // Transition to ON
      current_state = true;
      state_change_time = now;
      return true; // State changed
    }

    if (!new_detection && current_state) {
      // Only transition to OFF after minimum on-time
      if ((now - state_change_time) >= MIN_ON_TIME) {
        current_state = false;
        state_change_time = now;
        return true; // State changed
      }
    }

    return false; // No state change
  }
};
```

---

## 4. ESPHome Component Design

### 4.1 Component Structure Options

**Option 1: Pure YAML with Lambda (Recommended for Simplicity)**
- No custom C++ component needed
- Use existing `i2s_audio` microphone component
- Implement detection logic in lambda functions
- Suitable for RMS energy detection

**Option 2: Custom C++ Component (Recommended for Performance)**
- Full control over I2S and processing
- Efficient memory management
- Better for Goertzel/FFT implementations
- Packaged as external component

**Option 3: Hybrid Approach (Recommended)**
- Use native ESPHome I2S microphone
- Custom C++ component for signal processing
- YAML configuration for parameters
- Best balance of flexibility and performance

### 4.2 Directory Structure (Custom Component)

```
esphome-audio-sensor/
├── components/
│   └── beep_detector/
│       ├── __init__.py              # Python validation/registration
│       ├── beep_detector.h          # C++ component header
│       ├── beep_detector.cpp        # C++ implementation
│       ├── audio_processor.h        # Signal processing classes
│       └── audio_processor.cpp      # DSP algorithms
├── esphome-atom-d4d5d0.yaml        # Main config file
└── secrets.yaml                     # Wi-Fi credentials
```

### 4.3 YAML Configuration Design

**Complete Configuration Example:**

```yaml
esphome:
  name: esphome-web-d4d5d0
  friendly_name: m5Stack Echo d4d5d0
  min_version: 2025.9.0
  name_add_mac_suffix: false

esp32:
  variant: esp32
  framework:
    type: esp-idf

# Enable logging
logger:
  level: INFO

# Home Assistant integration
api:
  encryption:
    key: !secret api_encryption_key

ota:
  - platform: esphome
    password: !secret ota_password

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password

  # Fallback hotspot
  ap:
    ssid: "M5Atom-Beep-Sensor"
    password: !secret ap_password

# I2S Microphone Configuration
i2s_audio:
  - id: i2s_mic
    i2s_lrclk_pin: GPIO33
    i2s_bclk_pin: GPIO19

microphone:
  - platform: i2s_audio
    id: atom_mic
    i2s_audio_id: i2s_mic
    i2s_din_pin: GPIO22
    adc_type: external
    pdm: false
    sample_rate: 16000
    bits_per_sample: 16bit
    channel: left
    use_apll: false

# Custom Beep Detector Component
external_components:
  - source:
      type: local
      path: components
    components: [ beep_detector ]

# Beep Detection Configuration
beep_detector:
  id: water_heater_beep
  microphone_id: atom_mic

  # Signal Processing Parameters
  sample_rate: 16000
  buffer_size: 512

  # Detection Method: rms | goertzel | fft
  detection_method: goertzel

  # Frequency Configuration (for goertzel/fft)
  target_frequency: 2000      # Hz - adjust based on actual beep
  frequency_tolerance: 100     # Hz - ±100Hz window

  # Threshold Configuration
  energy_threshold: 1000       # RMS amplitude (0-32767)
  auto_threshold: true         # Dynamic threshold adjustment
  auto_threshold_percentile: 95  # Consider top 5% as beeps

  # Temporal Filtering
  min_consecutive_detections: 3  # Frames (3 * 32ms = 96ms)
  debounce_time: 200ms          # Minimum time between detections

  # State Hysteresis
  min_beep_duration: 100ms      # Minimum ON time
  min_silence_duration: 500ms   # Minimum OFF time

  # Performance Tuning
  update_interval: 32ms         # Match buffer size (512 / 16000)
  process_priority: 10          # Processing task priority

  # Diagnostics
  publish_energy_level: true    # Expose energy sensor
  publish_frequency: true       # Expose detected frequency

# Binary Sensor - Main Output
binary_sensor:
  - platform: beep_detector
    beep_detector_id: water_heater_beep
    name: "Water Heater Beeping"
    id: beep_sensor
    device_class: problem

    # Automations
    on_press:
      - logger.log: "⚠️ Water heater beep detected!"
      - homeassistant.event:
          event: esphome.water_heater_beep
          data:
            device_id: !lambda 'return App.get_name();'

    on_release:
      - logger.log: "✅ Water heater beep stopped"

# Diagnostic Sensors (optional)
sensor:
  - platform: beep_detector
    beep_detector_id: water_heater_beep
    type: energy
    name: "Audio Energy Level"
    id: audio_energy
    unit_of_measurement: "RMS"
    accuracy_decimals: 0
    update_interval: 1s

  - platform: beep_detector
    beep_detector_id: water_heater_beep
    type: frequency
    name: "Detected Frequency"
    id: detected_freq
    unit_of_measurement: "Hz"
    accuracy_decimals: 0
    update_interval: 1s
    filters:
      - or:
        - throttle: 5s      # Only update when changed
        - delta: 50.0       # Or frequency shifts >50Hz

  # System health
  - platform: wifi_signal
    name: "WiFi Signal"
    update_interval: 60s

  - platform: uptime
    name: "Uptime"
    update_interval: 60s

# Status LED (Atom Echo has RGB LED)
light:
  - platform: esp32_rmt_led_strip
    id: status_led
    name: "Status LED"
    pin: GPIO27
    num_leds: 1
    rmt_channel: 0
    rgb_order: GRB
    chipset: WS2812

    # Visual feedback
    effects:
      - pulse:
          name: "Beeping"
          transition_length: 0.5s
          update_interval: 0.5s

# Automation: Visual feedback on beep
script:
  - id: beep_visual_alert
    then:
      - light.turn_on:
          id: status_led
          brightness: 100%
          red: 100%
          green: 0%
          blue: 0%
          effect: "Beeping"
      - delay: 5s
      - light.turn_on:
          id: status_led
          brightness: 50%
          red: 0%
          green: 100%
          blue: 0%

# Configuration validation
interval:
  - interval: 60s
    then:
      - logger.log:
          format: "Beep detector running | Energy: %.0f | State: %s"
          args: [ 'id(audio_energy).state', 'id(beep_sensor).state ? "BEEP" : "OK"' ]
```

### 4.4 Configuration Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `microphone_id` | ID | required | Reference to I2S microphone |
| `sample_rate` | int | 16000 | Sampling frequency (Hz) |
| `buffer_size` | int | 512 | Samples per processing chunk |
| `detection_method` | enum | goertzel | rms, goertzel, fft |
| `target_frequency` | float | 2000 | Center frequency for detection (Hz) |
| `frequency_tolerance` | float | 100 | ±Hz range for frequency match |
| `energy_threshold` | float | 1000 | RMS amplitude threshold |
| `auto_threshold` | bool | false | Dynamic threshold adaptation |
| `min_consecutive_detections` | int | 3 | Frames before triggering |
| `debounce_time` | time | 200ms | Minimum time between events |
| `update_interval` | time | 32ms | Processing loop interval |

---

## 5. C++ Component Implementation

### 5.1 Component Class Header (`beep_detector.h`)

```cpp
#pragma once

#include "esphome/core/component.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/microphone/microphone.h"
#include <vector>
#include <cmath>

namespace esphome {
namespace beep_detector {

enum DetectionMethod {
  DETECTION_METHOD_RMS,
  DETECTION_METHOD_GOERTZEL,
  DETECTION_METHOD_FFT
};

class BeepDetectorComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::DATA; }

  // Configuration setters
  void set_microphone(microphone::Microphone *mic) { microphone_ = mic; }
  void set_sample_rate(uint32_t rate) { sample_rate_ = rate; }
  void set_buffer_size(size_t size) { buffer_size_ = size; }
  void set_detection_method(DetectionMethod method) { detection_method_ = method; }
  void set_target_frequency(float freq) { target_frequency_ = freq; }
  void set_frequency_tolerance(float tol) { frequency_tolerance_ = tol; }
  void set_energy_threshold(float threshold) { energy_threshold_ = threshold; }
  void set_min_consecutive_detections(uint8_t count) { min_consecutive_detections_ = count; }
  void set_debounce_time(uint32_t ms) { debounce_time_ms_ = ms; }

  // Sensor registration
  void set_binary_sensor(binary_sensor::BinarySensor *sensor) { binary_sensor_ = sensor; }
  void set_energy_sensor(sensor::Sensor *sensor) { energy_sensor_ = sensor; }
  void set_frequency_sensor(sensor::Sensor *sensor) { frequency_sensor_ = sensor; }

  // State accessors
  bool get_beep_state() const { return beep_detected_; }
  float get_current_energy() const { return current_energy_; }
  float get_current_frequency() const { return current_frequency_; }

 protected:
  // Audio processing
  void process_audio_buffer(const int16_t *samples, size_t count);
  float compute_rms(const int16_t *samples, size_t count);
  float compute_goertzel(const int16_t *samples, size_t count, float target_freq);
  void compute_fft(const int16_t *samples, size_t count);

  // Detection logic
  bool check_detection(float energy, float frequency);
  void update_state(bool detected);

  // DC removal filter
  void remove_dc_offset(int16_t *samples, size_t count);

  // Members
  microphone::Microphone *microphone_{nullptr};
  binary_sensor::BinarySensor *binary_sensor_{nullptr};
  sensor::Sensor *energy_sensor_{nullptr};
  sensor::Sensor *frequency_sensor_{nullptr};

  // Configuration
  uint32_t sample_rate_{16000};
  size_t buffer_size_{512};
  DetectionMethod detection_method_{DETECTION_METHOD_GOERTZEL};
  float target_frequency_{2000.0f};
  float frequency_tolerance_{100.0f};
  float energy_threshold_{1000.0f};
  uint8_t min_consecutive_detections_{3};
  uint32_t debounce_time_ms_{200};

  // State
  std::vector<int16_t> audio_buffer_;
  bool beep_detected_{false};
  float current_energy_{0.0f};
  float current_frequency_{0.0f};
  uint8_t consecutive_detections_{0};
  uint32_t last_detection_time_{0};
  uint32_t state_change_time_{0};

  // DC filter state
  float dc_estimate_{0.0f};
  static constexpr float DC_ALPHA = 0.995f;

  // Goertzel state
  float goertzel_coeff_{0.0f};
  float goertzel_q1_{0.0f};
  float goertzel_q2_{0.0f};
};

}  // namespace beep_detector
}  // namespace esphome
```

### 5.2 Key Implementation Functions

**Goertzel Algorithm Implementation:**

```cpp
float BeepDetectorComponent::compute_goertzel(const int16_t *samples,
                                               size_t count,
                                               float target_freq) {
  // Pre-compute coefficient
  float k = (float)(count * target_freq) / sample_rate_;
  float omega = (2.0f * M_PI * k) / count;
  float coeff = 2.0f * cos(omega);

  float q0, q1 = 0.0f, q2 = 0.0f;

  // Process samples
  for (size_t i = 0; i < count; i++) {
    q0 = coeff * q1 - q2 + (float)samples[i];
    q2 = q1;
    q1 = q0;
  }

  // Compute magnitude
  float magnitude = sqrt(q1 * q1 + q2 * q2 - q1 * q2 * coeff);

  // Normalize by buffer size
  return magnitude / count;
}
```

**Detection Logic:**

```cpp
bool BeepDetectorComponent::check_detection(float energy, float frequency) {
  bool criteria_met = false;

  // Energy check
  if (energy > energy_threshold_) {

    // Frequency check (for Goertzel/FFT)
    if (detection_method_ == DETECTION_METHOD_RMS) {
      criteria_met = true;  // No frequency check for RMS
    } else {
      float freq_min = target_frequency_ - frequency_tolerance_;
      float freq_max = target_frequency_ + frequency_tolerance_;

      if (frequency >= freq_min && frequency <= freq_max) {
        criteria_met = true;
      }
    }
  }

  // Consecutive detection counter
  if (criteria_met) {
    consecutive_detections_++;
  } else {
    consecutive_detections_ = max(0, consecutive_detections_ - 1);
  }

  // Threshold check
  if (consecutive_detections_ >= min_consecutive_detections_) {
    uint32_t now = millis();
    if ((now - last_detection_time_) > debounce_time_ms_) {
      last_detection_time_ = now;
      return true;
    }
  }

  return false;
}
```

---

## 6. Home Assistant Integration

### 6.1 Entity Exposure

**Automatically Created Entities:**

```yaml
binary_sensor:
  - platform: esphome
    name: "Water Heater Beeping"
    unique_id: "m5atom_d4d5d0_beep_sensor"
    device_class: problem
    entity_id: binary_sensor.water_heater_beeping

sensor:
  - platform: esphome
    name: "Audio Energy Level"
    unique_id: "m5atom_d4d5d0_audio_energy"
    unit_of_measurement: "RMS"
    state_class: measurement
    entity_id: sensor.water_heater_audio_energy

  - platform: esphome
    name: "Detected Frequency"
    unique_id: "m5atom_d4d5d0_detected_freq"
    unit_of_measurement: "Hz"
    state_class: measurement
    entity_id: sensor.water_heater_detected_frequency
```

### 6.2 Automation Examples

**Home Assistant Automation (YAML):**

```yaml
automation:
  - alias: "Water Heater Beep Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.water_heater_beeping
        to: "on"
    condition:
      - condition: time
        after: "06:00:00"
        before: "23:00:00"
    action:
      - service: notify.mobile_app
        data:
          title: "🚨 Water Heater Alert"
          message: "Water heater is beeping - check for error!"
          data:
            priority: high
            sound: alarm.mp3

      - service: light.turn_on
        target:
          entity_id: light.living_room
        data:
          brightness: 255
          rgb_color: [255, 0, 0]
          flash: long

  - alias: "Water Heater Beep Resolved"
    trigger:
      - platform: state
        entity_id: binary_sensor.water_heater_beeping
        to: "off"
        for: "00:02:00"
    action:
      - service: notify.mobile_app
        data:
          title: "✅ Water Heater Normal"
          message: "Beeping has stopped"
```

### 6.3 Lovelace Dashboard Card

```yaml
type: entities
title: Water Heater Monitor
entities:
  - entity: binary_sensor.water_heater_beeping
    name: Status
    icon: mdi:water-boiler
  - entity: sensor.water_heater_audio_energy
    name: Audio Level
    icon: mdi:waveform
  - entity: sensor.water_heater_detected_frequency
    name: Frequency
    icon: mdi:sine-wave
  - entity: sensor.wifi_signal_m5atom
    name: Signal Strength
  - entity: sensor.uptime_m5atom
    name: Uptime
```

---

## 7. Performance Analysis

### 7.1 CPU Budget (ESP32 @ 240MHz)

| Stage | Operations | Time (ms) | CPU % |
|-------|-----------|-----------|-------|
| I2S Read (512 samples) | DMA transfer | 0.1 | <1% |
| DC Removal | 512 ops | 0.05 | <1% |
| Preprocessing | 1024 ops | 0.1 | <1% |
| Goertzel (single freq) | ~1500 ops | 0.5 | 2% |
| FFT (512-point) | ~20K ops | 15 | 45% |
| RMS Energy | 512 ops | 0.05 | <1% |
| Detection Logic | 100 ops | 0.02 | <1% |
| State Update | 50 ops | 0.01 | <1% |
| **Total (Goertzel)** | | **~1ms** | **~5%** |
| **Total (FFT)** | | **~16ms** | **~50%** |
| **Total (RMS only)** | | **~0.3ms** | **~1%** |

**Buffer Interval:** 512 samples @ 16kHz = 32ms
**Processing Headroom:** Goertzel = 31ms spare, FFT = 16ms spare

### 7.2 Memory Budget

| Component | Size | Notes |
|-----------|------|-------|
| Audio buffer (int16) | 1024 B | 512 samples × 2 bytes |
| FFT working buffer | 4096 B | Float32 × 2 (in/out) |
| Goertzel state | 16 B | 4 floats |
| Detection state | 32 B | Counters, timestamps |
| Component overhead | 128 B | Class members |
| **Total (RMS/Goertzel)** | **1.2 KB** | Minimal |
| **Total (FFT)** | **5.3 KB** | Larger footprint |

**ESP32 SRAM:** 520 KB total → <2% usage for audio processing

### 7.3 Latency Analysis

```
Physical beep start
  ↓ (0-32ms)        ← Capture buffer fill time
Audio buffer ready
  ↓ (0.3-16ms)      ← Processing time (method dependent)
Detection complete
  ↓ (96-200ms)      ← Consecutive detections + debounce
Binary sensor update
  ↓ (~50ms)         ← ESPHome API update
Home Assistant receives event
  ↓ (~100ms)        ← HA automation processing
Action triggered

Total latency: 250-500ms (acceptable for alarm systems)
```

---

## 8. Calibration & Tuning Guide

### 8.1 Initial Setup Procedure

**Step 1: Baseline Audio Capture**
```yaml
# Temporary config to capture ambient noise
logger:
  level: DEBUG
  logs:
    beep_detector: VERY_VERBOSE

sensor:
  - platform: beep_detector
    type: energy
    name: "Audio Energy"

# Monitor for 1 hour, note typical values
```

**Step 2: Frequency Identification**

Option A: Use diagnostic frequency sensor
```yaml
# Enable frequency output
beep_detector:
  publish_frequency: true
  detection_method: fft  # Broadband analysis

# Trigger actual beep, check logs for peak frequency
```

Option B: Analyze recorded audio
```bash
# Record audio from microphone
# Analyze with Audacity or Python:
import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

rate, data = wavfile.read('beep_sample.wav')
frequencies, power = welch(data, rate, nperseg=1024)
peak_freq = frequencies[np.argmax(power)]
print(f"Beep frequency: {peak_freq:.0f} Hz")
```

**Step 3: Threshold Tuning**
```yaml
# Start conservative
beep_detector:
  energy_threshold: 5000  # High threshold
  min_consecutive_detections: 5

# Gradually reduce until reliable detection
# Monitor false positive rate
```

### 8.2 Tuning Parameters

**Sensitivity vs. False Positives:**

| Scenario | energy_threshold | min_consecutive | frequency_tolerance |
|----------|------------------|-----------------|---------------------|
| Very quiet room | 500 | 2 | 50 Hz |
| Typical home | 1000 | 3 | 100 Hz |
| Noisy environment | 3000 | 5 | 150 Hz |
| Industrial | 8000 | 7 | 200 Hz |

**Detection Speed vs. Reliability:**

| Priority | buffer_size | min_consecutive | debounce_time |
|----------|-------------|-----------------|---------------|
| Fast response | 256 | 2 | 100ms |
| Balanced | 512 | 3 | 200ms |
| Reliable | 1024 | 5 | 500ms |

---

## 9. Testing & Validation

### 9.1 Unit Test Cases

**Test 1: Silent Environment**
- Input: No audio (ambient <20dB)
- Expected: binary_sensor = OFF, energy < 100

**Test 2: Target Beep**
- Input: Beep @ target frequency, >threshold
- Expected: binary_sensor = ON within 500ms

**Test 3: Off-Frequency Tone**
- Input: Continuous tone @ target_freq ± 500Hz
- Expected: binary_sensor = OFF (frequency rejection)

**Test 4: Brief Transient**
- Input: Single loud click (<50ms)
- Expected: binary_sensor = OFF (duration filter)

**Test 5: Repeated Beeps**
- Input: 5 beeps, 1s interval
- Expected: 5 ON events, proper debouncing

**Test 6: Gradual Volume Change**
- Input: Beep ramping from 0% → 100% over 2s
- Expected: ON when crosses threshold

### 9.2 Integration Test Scenarios

**Scenario A: Morning Routine**
- Background: Conversation, coffee machine, door closing
- Trigger: Water heater beep @ 07:30
- Validate: Detection within 1s, HA notification received

**Scenario B: TV/Music Playing**
- Background: TV at normal volume (65dB)
- Trigger: Water heater beep
- Validate: Detection regardless of background noise

**Scenario C: Overnight Monitoring**
- Duration: 8 hours
- Validate: No false positives, <0.01% false alarm rate

**Scenario D: Wi-Fi Dropout**
- Test: Disconnect Wi-Fi during beep
- Validate: Detection persists, event queued, delivered on reconnect

### 9.3 Performance Benchmarks

**Acceptance Criteria:**
- ✅ Detection latency: <500ms from beep start
- ✅ False positive rate: <1 per 24 hours
- ✅ False negative rate: <1%
- ✅ CPU usage: <10% average
- ✅ Memory usage: <10KB
- ✅ Power consumption: <500mW (USB powered, not critical)
- ✅ Wi-Fi reliability: >99.9% uptime

---

## 10. Deployment & Maintenance

### 10.1 Installation Procedure

1. **Flash ESPHome firmware**
   ```bash
   esphome run esphome-atom-d4d5d0.yaml
   ```

2. **Physical placement**
   - Within 3-5 meters of water heater
   - Avoid direct airflow (HVAC vents)
   - USB power source (5V, 500mA minimum)

3. **Initial calibration**
   - Monitor ambient noise for 1 hour
   - Trigger test beep, verify detection
   - Adjust thresholds if needed

4. **Home Assistant setup**
   - Verify entity appearance
   - Create automations
   - Test notification delivery

### 10.2 Monitoring & Diagnostics

**Health Checks:**
```yaml
# Add to Home Assistant
sensor:
  - platform: history_stats
    name: "Beep Events Today"
    entity_id: binary_sensor.water_heater_beeping
    state: "on"
    type: count
    start: "{{ now().replace(hour=0, minute=0, second=0) }}"
    end: "{{ now() }}"

binary_sensor:
  - platform: template
    sensors:
      beep_detector_healthy:
        friendly_name: "Beep Detector Health"
        value_template: >
          {{ (as_timestamp(now()) - as_timestamp(states.sensor.uptime_m5atom.last_changed)) < 300 }}
        device_class: problem
```

**Alert on Malfunction:**
```yaml
automation:
  - alias: "Beep Detector Offline"
    trigger:
      - platform: state
        entity_id: binary_sensor.beep_detector_healthy
        to: "off"
        for: "00:05:00"
    action:
      - service: notify.admin
        data:
          message: "M5Atom beep detector is offline!"
```

### 10.3 Update Strategy

**Firmware Updates:**
- OTA updates via ESPHome dashboard
- Test on development device first
- Backup current config before updating
- Monitor for 24h after update

**Configuration Changes:**
- Version control YAML in git
- Document threshold changes
- A/B testing for major changes

---

## 11. Troubleshooting Guide

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| No detection ever | Threshold too high | Lower `energy_threshold` by 50% |
| | Frequency mismatch | Use FFT to identify actual frequency |
| | Microphone failure | Check I2S wiring, test with audio sensor |
| Constant false positives | Threshold too low | Increase `energy_threshold` |
| | Broad frequency range | Reduce `frequency_tolerance` |
| | HVAC noise | Add high-pass filter, increase min_consecutive |
| Intermittent detection | Weak Wi-Fi signal | Relocate device, add Wi-Fi extender |
| | Power supply issues | Use quality USB power adapter |
| | Buffer overflow | Reduce processing load (use RMS not FFT) |
| Delayed detection (>1s) | Too many consecutive hits | Reduce `min_consecutive_detections` |
| | High debounce time | Lower `debounce_time` |
| High CPU usage | FFT enabled | Switch to Goertzel or RMS method |
| | Sample rate too high | Use 16kHz, not 44.1kHz |
| Random restarts | Memory overflow | Check buffer sizes, reduce FFT use |
| | Watchdog timeout | Reduce processing complexity |

---

## 12. Future Enhancements

### 12.1 Phase 2 Features

**Machine Learning Integration:**
- Use TensorFlow Lite for Microcontrollers
- Train custom beep classifier model
- On-device inference (<10ms)
- Improved accuracy and pattern recognition

**Multi-Pattern Support:**
- Detect different beep patterns (error vs. alert)
- Classify by urgency (fast vs. slow beeping)
- Multiple appliances (water heater, smoke alarm, etc.)

**Cloud Integration:**
- Historical beep event logging
- Pattern analysis over time
- Predictive maintenance alerts

### 12.2 Hardware Upgrades

**Audio Enhancements:**
- External microphone for better placement
- Stereo capture for directionality
- Higher SNR microphone (>70dB)

**Connectivity:**
- PoE power for remote locations
- Ethernet for ultra-reliable connection
- Cellular backup for critical deployments

---

## 13. References & Resources

### 13.1 Technical Documentation

**ESPHome:**
- [I2S Audio Component](https://esphome.io/components/i2s_audio.html)
- [Binary Sensor](https://esphome.io/components/binary_sensor/index.html)
- [Custom Components](https://esphome.io/custom/custom_component.html)

**ESP32:**
- [I2S Driver API](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html)
- [ESP-DSP Library](https://github.com/espressif/esp-dsp)

**Signal Processing:**
- Goertzel Algorithm: [Wikipedia](https://en.wikipedia.org/wiki/Goertzel_algorithm)
- FFT Implementation: [CMSIS-DSP](https://arm-software.github.io/CMSIS_5/DSP/html/index.html)

### 13.2 Similar Projects

- [ESPHome Voice Assistant](https://esphome.io/components/voice_assistant.html)
- [ESP32 Audio Reactive](https://github.com/atuline/WLED-audio-reactive-LED-strip)
- [OpenMQTTGateway Audio](https://docs.openmqttgateway.com/)

---

## Appendix A: Complete Component Code Template

### `components/beep_detector/__init__.py`

```python
import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, sensor, microphone
from esphome.const import (
    CONF_ID,
    CONF_MICROPHONE_ID,
    UNIT_HERTZ,
    ICON_WAVEFORM,
)

DEPENDENCIES = ["microphone"]
AUTO_LOAD = ["binary_sensor", "sensor"]

beep_detector_ns = cg.esphome_ns.namespace("beep_detector")
BeepDetectorComponent = beep_detector_ns.class_("BeepDetectorComponent", cg.Component)

CONF_SAMPLE_RATE = "sample_rate"
CONF_BUFFER_SIZE = "buffer_size"
CONF_DETECTION_METHOD = "detection_method"
CONF_TARGET_FREQUENCY = "target_frequency"
CONF_FREQUENCY_TOLERANCE = "frequency_tolerance"
CONF_ENERGY_THRESHOLD = "energy_threshold"
CONF_MIN_CONSECUTIVE = "min_consecutive_detections"
CONF_DEBOUNCE_TIME = "debounce_time"

DETECTION_METHODS = {
    "rms": 0,
    "goertzel": 1,
    "fft": 2,
}

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(BeepDetectorComponent),
    cv.Required(CONF_MICROPHONE_ID): cv.use_id(microphone.Microphone),
    cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.int_range(min=8000, max=48000),
    cv.Optional(CONF_BUFFER_SIZE, default=512): cv.int_range(min=128, max=2048),
    cv.Optional(CONF_DETECTION_METHOD, default="goertzel"): cv.enum(DETECTION_METHODS),
    cv.Optional(CONF_TARGET_FREQUENCY, default=2000.0): cv.float_range(min=100.0, max=10000.0),
    cv.Optional(CONF_FREQUENCY_TOLERANCE, default=100.0): cv.float_range(min=10.0, max=1000.0),
    cv.Optional(CONF_ENERGY_THRESHOLD, default=1000.0): cv.float_range(min=0.0),
    cv.Optional(CONF_MIN_CONSECUTIVE, default=3): cv.int_range(min=1, max=10),
    cv.Optional(CONF_DEBOUNCE_TIME, default="200ms"): cv.positive_time_period_milliseconds,
}).extend(cv.COMPONENT_SCHEMA)

async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE_ID])
    cg.add(var.set_microphone(mic))

    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_buffer_size(config[CONF_BUFFER_SIZE]))
    cg.add(var.set_detection_method(config[CONF_DETECTION_METHOD]))
    cg.add(var.set_target_frequency(config[CONF_TARGET_FREQUENCY]))
    cg.add(var.set_frequency_tolerance(config[CONF_FREQUENCY_TOLERANCE]))
    cg.add(var.set_energy_threshold(config[CONF_ENERGY_THRESHOLD]))
    cg.add(var.set_min_consecutive_detections(config[CONF_MIN_CONSECUTIVE]))
    cg.add(var.set_debounce_time(config[CONF_DEBOUNCE_TIME]))
```

---

## Appendix B: Testing Checklist

- [ ] I2S microphone captures audio at 16kHz
- [ ] Audio buffer fills without overflow
- [ ] DC offset removal functioning
- [ ] RMS energy calculation correct
- [ ] Goertzel algorithm detects target frequency
- [ ] Detection threshold properly configured
- [ ] Consecutive detection counter works
- [ ] Debouncing prevents rapid toggling
- [ ] Binary sensor updates in Home Assistant
- [ ] Energy sensor publishes values
- [ ] Frequency sensor shows detected tone
- [ ] Wi-Fi connection stable
- [ ] OTA updates successful
- [ ] Power consumption acceptable
- [ ] CPU usage within budget
- [ ] No memory leaks over 24h
- [ ] False positive rate <1/day
- [ ] Detection latency <500ms
- [ ] Automations trigger correctly
- [ ] Logging provides useful diagnostics

---

## Document Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-10 | Architecture Team | Initial comprehensive design |

---

**END OF ARCHITECTURE DOCUMENT**
