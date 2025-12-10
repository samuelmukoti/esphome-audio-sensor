# Implementation Checklist
## ESPHome Beep Detector - Step-by-Step Guide

---

## Pre-Implementation Phase

### ☐ Audio Analysis
- [ ] Extract audio file characteristics (done: 48kHz stereo, 18.5s duration)
- [ ] Convert to mono 16kHz for analysis
- [ ] Identify beep frequency using FFT/spectrum analyzer
- [ ] Measure beep duration and pattern
- [ ] Determine amplitude range (dB/RMS)
- [ ] Identify background noise characteristics
- [ ] Document findings in analysis report

**Tools needed:**
```bash
# Convert audio for analysis
ffmpeg -i water_heater_beeping_error_sound.m4a \
       -ar 16000 -ac 1 beep_16k_mono.wav

# Python analysis script
python3 <<EOF
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt

# Load audio
rate, data = wavfile.read('beep_16k_mono.wav')

# Compute spectrogram
f, t, Sxx = signal.spectrogram(data, rate, nperseg=1024)

# Find peak frequency
peak_freq_idx = np.argmax(np.mean(Sxx, axis=1))
peak_frequency = f[peak_freq_idx]

print(f"Sample rate: {rate} Hz")
print(f"Duration: {len(data)/rate:.2f} seconds")
print(f"Peak frequency: {peak_frequency:.0f} Hz")
print(f"RMS amplitude: {np.sqrt(np.mean(data**2)):.0f}")

# Plot
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Beep Spectrogram')
plt.colorbar(label='Power [dB]')
plt.ylim(0, 4000)
plt.savefig('beep_analysis.png')
plt.show()
EOF
```

### ☐ Method Selection
- [ ] Review audio analysis results
- [ ] Choose detection method (RMS / Goertzel / FFT)
- [ ] Document decision rationale
- [ ] Estimate development effort
- [ ] Plan fallback approach if primary fails

**Decision Matrix:**
```
If beep frequency is KNOWN and STABLE → Goertzel
If beep frequency is UNKNOWN → FFT (then switch to Goertzel)
If beep is ANY loud sound → RMS
If prototype/MVP needed fast → RMS → iterate to Goertzel
```

### ☐ Development Environment Setup
- [ ] Install ESPHome CLI (`pip install esphome`)
- [ ] Verify ESP32 toolchain (`esphome version`)
- [ ] Set up Home Assistant test instance
- [ ] Prepare M5Stack Atom Echo hardware
- [ ] Test USB connection and serial access
- [ ] Create development branch (if using git)
- [ ] Backup existing configuration

---

## Phase 1: Basic Audio Capture (MVP)

### ☐ I2S Microphone Configuration
- [ ] Add I2S audio platform to YAML
- [ ] Configure GPIO pins (BCLK=19, LRCLK=33, DIN=22)
- [ ] Set sample rate (16000 Hz)
- [ ] Set bit depth (16-bit)
- [ ] Configure channel (left/mono)
- [ ] Flash and verify I2S initialization in logs

**YAML snippet:**
```yaml
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
```

### ☐ Audio Capture Verification
- [ ] Enable debug logging for I2S
- [ ] Flash firmware to device
- [ ] Monitor logs for I2S initialization
- [ ] Verify DMA buffer allocation
- [ ] Check for I2S read errors
- [ ] Confirm audio data is flowing

**Validation:**
```yaml
logger:
  level: DEBUG
  logs:
    i2s: VERY_VERBOSE
    microphone: VERY_VERBOSE
```

### ☐ Simple Energy Monitoring (First Milestone)
- [ ] Add interval component to read audio
- [ ] Implement basic RMS calculation in lambda
- [ ] Create sensor to display energy level
- [ ] Flash and verify energy readings
- [ ] Test with different noise levels
- [ ] Document typical ambient energy values

**MVP YAML:**
```yaml
sensor:
  - platform: template
    name: "Audio Energy"
    id: audio_energy
    unit_of_measurement: "RMS"
    update_interval: 100ms
    lambda: |-
      // Placeholder - implement I2S read and RMS calc
      return 0;
```

**Success Criteria:**
- ✅ Logs show I2S reading audio
- ✅ Energy sensor updates regularly
- ✅ Values change with ambient noise
- ✅ No crashes or errors for 10 minutes

---

## Phase 2: Detection Logic Implementation

### ☐ Choose Implementation Path

**Path A: Pure YAML (RMS only)**
- [ ] Implement RMS in lambda function
- [ ] Add threshold comparison
- [ ] Create template binary sensor
- [ ] Test detection with loud sounds
- [ ] Skip to Phase 3

**Path B: Custom Component (Goertzel/FFT)**
- [ ] Continue with steps below

### ☐ Component Structure Setup
- [ ] Create `components/beep_detector/` directory
- [ ] Create `__init__.py` (Python config validation)
- [ ] Create `beep_detector.h` (C++ header)
- [ ] Create `beep_detector.cpp` (C++ implementation)
- [ ] Add component to external_components in YAML

**Directory structure:**
```
components/
└── beep_detector/
    ├── __init__.py
    ├── beep_detector.h
    ├── beep_detector.cpp
    └── audio_processor.cpp (optional)
```

### ☐ Component Skeleton
- [ ] Define BeepDetectorComponent class
- [ ] Implement setup() method
- [ ] Implement loop() method
- [ ] Add configuration setters
- [ ] Add binary sensor registration
- [ ] Verify component compiles

**Minimal header (`beep_detector.h`):**
```cpp
#pragma once

#include "esphome/core/component.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/microphone/microphone.h"

namespace esphome {
namespace beep_detector {

class BeepDetectorComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override;

  void set_microphone(microphone::Microphone *mic);
  void set_binary_sensor(binary_sensor::BinarySensor *sensor);

 protected:
  microphone::Microphone *microphone_{nullptr};
  binary_sensor::BinarySensor *binary_sensor_{nullptr};
};

}  // namespace beep_detector
}  // namespace esphome
```

### ☐ Audio Processing Implementation
- [ ] Add I2S buffer reading code
- [ ] Implement DC offset removal
- [ ] Add high-pass filter (optional)
- [ ] Implement chosen algorithm (RMS/Goertzel/FFT)
- [ ] Add unit tests for algorithms (optional)
- [ ] Verify processing performance (<10ms per buffer)

**Implementation priority:**
1. Basic RMS calculation (simplest)
2. Goertzel for target frequency
3. FFT spectrum analysis (if needed)

### ☐ Detection Logic
- [ ] Implement threshold comparison
- [ ] Add consecutive detection counter
- [ ] Implement debouncing logic
- [ ] Add state machine (IDLE/DETECTING/ACTIVE)
- [ ] Implement hysteresis
- [ ] Test with various scenarios

**Test scenarios:**
- Silent environment (should be OFF)
- Single loud transient (should ignore)
- Sustained beep (should detect ON)
- Beep stops (should transition to OFF)

### ☐ Configuration Interface
- [ ] Define config schema in `__init__.py`
- [ ] Add validation for parameters
- [ ] Implement config setters in C++
- [ ] Document all parameters in YAML
- [ ] Test with different config values

**Key parameters:**
```yaml
beep_detector:
  sample_rate: 16000
  buffer_size: 512
  detection_method: goertzel  # or rms or fft
  target_frequency: 2000
  energy_threshold: 1000
  min_consecutive_detections: 3
  debounce_time: 200ms
```

---

## Phase 3: Testing & Calibration

### ☐ Unit Testing
- [ ] Test silent environment (no false positives)
- [ ] Test target beep (correct detection)
- [ ] Test off-frequency sounds (rejection)
- [ ] Test brief transients (filtering)
- [ ] Test repeated beeps (debouncing)
- [ ] Test volume variations (threshold)

**Test log template:**
```
Test: Silent Environment
Expected: binary_sensor = OFF, energy < 100
Actual: ___________
Result: PASS / FAIL
Notes: ___________

Test: Target Beep @ 2000 Hz
Expected: binary_sensor = ON within 500ms
Actual: ___________
Result: PASS / FAIL
Notes: ___________
```

### ☐ Threshold Calibration
- [ ] Record ambient baseline (1 hour)
- [ ] Note min/max/avg energy values
- [ ] Set initial threshold = avg × 5
- [ ] Trigger test beep, verify detection
- [ ] Adjust threshold if needed
- [ ] Document final threshold value

**Calibration worksheet:**
```
Ambient Energy (1 hour monitoring):
  Min: _____ RMS
  Max: _____ RMS
  Avg: _____ RMS

Test Beep Energy:
  Peak: _____ RMS
  Avg during beep: _____ RMS

Recommended threshold: _____ RMS
Frequency detected: _____ Hz
```

### ☐ Integration Testing
- [ ] Deploy in actual location (near water heater)
- [ ] Monitor for 24 hours
- [ ] Count false positives
- [ ] Count missed detections (false negatives)
- [ ] Check Wi-Fi stability
- [ ] Verify Home Assistant integration
- [ ] Test automation triggers

**24-Hour Test Results:**
```
Duration: 24 hours
False Positives: _____ (target: <1)
False Negatives: _____ (target: 0)
Wi-Fi Disconnects: _____ (target: 0)
Average CPU: _____% (target: <10%)
Peak Memory: _____KB (target: <20KB)
```

### ☐ Performance Validation
- [ ] Measure detection latency (beep start → sensor ON)
- [ ] Monitor CPU usage over time
- [ ] Check memory usage (no leaks)
- [ ] Verify processing time per buffer
- [ ] Test under Wi-Fi congestion
- [ ] Verify OTA update works

**Performance benchmarks:**
```
Detection Latency: _____ ms (target: <500ms)
CPU Usage (avg): _____% (target: <10%)
CPU Usage (peak): _____% (target: <50%)
Memory Usage: _____ KB (target: <20KB)
Buffer Processing: _____ ms (target: <32ms)
```

---

## Phase 4: Home Assistant Integration

### ☐ Entity Configuration
- [ ] Verify binary sensor appears in HA
- [ ] Set friendly name and icon
- [ ] Configure device class (problem/safety)
- [ ] Add to appropriate area/room
- [ ] Create Lovelace card
- [ ] Test manual state display

### ☐ Automation Setup
- [ ] Create notification automation (beep detected)
- [ ] Add time-of-day conditions
- [ ] Configure notification service (mobile app/etc)
- [ ] Add visual alert (lights, etc)
- [ ] Create resolution notification (beep stopped)
- [ ] Test automation end-to-end

**Example automation:**
```yaml
automation:
  - alias: "Water Heater Beep Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.water_heater_beeping
        to: "on"
    action:
      - service: notify.mobile_app
        data:
          title: "Water Heater Alert"
          message: "Beeping detected - check error!"
```

### ☐ Diagnostic Dashboard
- [ ] Create sensors card (energy, frequency)
- [ ] Add history graph
- [ ] Add event log
- [ ] Add system health (WiFi, uptime)
- [ ] Add manual calibration controls (optional)

---

## Phase 5: Production Deployment

### ☐ Final Configuration
- [ ] Set optimal thresholds from testing
- [ ] Disable verbose logging
- [ ] Enable encryption (API key)
- [ ] Set OTA password
- [ ] Configure fallback AP
- [ ] Backup final configuration to git

### ☐ Physical Installation
- [ ] Mount device near water heater (2-5m)
- [ ] Verify stable Wi-Fi signal
- [ ] Connect reliable USB power
- [ ] Test detection from installation location
- [ ] Verify LED is visible for status
- [ ] Document installation location

**Installation checklist:**
```
Location: _____________________
Distance from beep source: _____ meters
Wi-Fi signal strength: _____ dBm
Power source: _____________________
Accessible for maintenance: YES / NO
LED visible: YES / NO
```

### ☐ Monitoring Setup
- [ ] Create health check automation
- [ ] Add offline alert (>5 min no heartbeat)
- [ ] Add daily summary (beep count)
- [ ] Configure log retention
- [ ] Set up metric tracking (optional)

### ☐ Documentation
- [ ] Document final configuration
- [ ] Document calibration values
- [ ] Create user guide for family members
- [ ] Document troubleshooting steps
- [ ] Create maintenance schedule

**User guide topics:**
- What the sensor does
- How to identify false alerts
- When to recalibrate
- How to check if working
- Who to contact for issues

---

## Phase 6: Maintenance & Optimization

### ☐ Routine Monitoring (Weekly)
- [ ] Check false positive rate
- [ ] Verify detection latency acceptable
- [ ] Review automation logs
- [ ] Check device uptime
- [ ] Verify Wi-Fi signal stable

### ☐ Periodic Recalibration (Monthly)
- [ ] Test with actual beep
- [ ] Review threshold effectiveness
- [ ] Adjust if environment changed
- [ ] Update configuration if needed
- [ ] Document calibration changes

### ☐ Firmware Updates (As Needed)
- [ ] Review ESPHome release notes
- [ ] Test update on dev device first
- [ ] Backup current configuration
- [ ] Schedule update during low-risk time
- [ ] Monitor for 24h post-update
- [ ] Roll back if issues detected

---

## Completion Criteria

### ✅ MVP Complete
- [ ] Audio capture working
- [ ] Basic detection functioning
- [ ] Binary sensor in Home Assistant
- [ ] Simple automation working

### ✅ Production Ready
- [ ] False positive rate <1 per 24h
- [ ] False negative rate <1%
- [ ] Detection latency <500ms
- [ ] 24-hour stability test passed
- [ ] Home Assistant integration complete
- [ ] Documentation complete

### ✅ Optimized
- [ ] CPU usage <10% average
- [ ] Frequency-selective detection working
- [ ] Adaptive thresholding enabled (optional)
- [ ] Pattern recognition implemented (optional)
- [ ] Machine learning integration (future)

---

## Troubleshooting Reference

### Issue: No I2S audio
**Check:**
- GPIO pins correct (19, 33, 22)
- I2S platform enabled
- Microphone platform configured
- Logs show initialization

**Fix:**
- Verify hardware connections
- Check logs for errors
- Try different sample rate
- Verify ESP-IDF framework

### Issue: Constant false positives
**Check:**
- Threshold too low
- Frequency tolerance too wide
- Background noise level

**Fix:**
- Increase energy_threshold
- Narrow frequency_tolerance
- Increase min_consecutive_detections
- Add time-of-day filtering

### Issue: No detection
**Check:**
- Threshold too high
- Wrong target frequency
- Microphone not working
- Processing errors in logs

**Fix:**
- Lower threshold by 50%
- Analyze audio to find actual frequency
- Test microphone with energy sensor
- Check logs for exceptions

### Issue: High CPU usage
**Check:**
- Detection method (FFT = high)
- Sample rate (higher = more CPU)
- Buffer size (smaller = more frequent)

**Fix:**
- Switch to Goertzel or RMS
- Reduce sample rate to 8kHz
- Increase buffer size to 1024

---

## Success Metrics

### Technical Metrics
- Detection accuracy: >99%
- Latency: <500ms
- CPU usage: <10%
- Memory usage: <20KB
- Uptime: >99.9%

### User Experience Metrics
- False alarms per week: <1
- Missed alerts per month: <1
- Time to notification: <1 second
- User satisfaction: High

### Operational Metrics
- Wi-Fi stability: >99.9%
- Power consumption: <500mW
- OTA success rate: 100%
- Mean time between recalibration: >30 days

---

## Final Checklist

- [ ] All phases completed
- [ ] Production deployment successful
- [ ] 7-day stability test passed
- [ ] User training completed
- [ ] Documentation finalized
- [ ] Monitoring configured
- [ ] Backup procedures established
- [ ] Maintenance schedule created

**Project Status: ____________**
**Sign-off Date: ____________**
**Notes: ____________________**

---

**End of Implementation Checklist**
