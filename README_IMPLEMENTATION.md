# Beep Detector Implementation Guide

## Overview

This implementation provides a real-time beep detection system for the M5Stack Atom Echo device using ESPHome. It uses the Goertzel algorithm to efficiently detect a specific frequency (2,615 Hz beep tone) with minimal CPU overhead.

## Architecture

### Components

1. **Custom C++ Component** (`custom_components/beep_detector/`)
   - `beep_detector.h`: Component interface and configuration
   - `beep_detector.cpp`: Goertzel algorithm implementation and state machine
   - `__init__.py`: ESPHome component registration and YAML schema

2. **Detection Algorithm**
   - **Goertzel Algorithm**: Efficient single-frequency DFT for 2,615 Hz detection
   - **Multi-Criteria Detection**: Both frequency energy AND RMS amplitude must exceed thresholds
   - **State Machine**: IDLE → DETECTING → CONFIRMED → COOLDOWN
   - **Debouncing**: Requires consecutive detections to confirm (reduces false positives)
   - **Duration Validation**: Beep must last 40-100ms to be valid

3. **Hardware Configuration**
   - M5Stack Atom Echo with SPM1423 PDM microphone
   - I2S interface on GPIO19/23/33
   - 16 kHz sample rate, 16-bit samples
   - Left channel audio input

## File Structure

```
/Users/sam/tmp/esphome-audio-sensor/
├── esphome-atom-d4d5d0.yaml           # Main ESPHome configuration
├── custom_components/
│   └── beep_detector/
│       ├── beep_detector.h             # Component header
│       ├── beep_detector.cpp           # Implementation
│       └── __init__.py                 # ESPHome integration
└── README_IMPLEMENTATION.md            # This file
```

## Configuration Parameters

### Detection Parameters (in YAML)

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `target_frequency` | 2615.0 | 100-8000 Hz | Primary beep frequency to detect |
| `sample_rate` | 16000 | 8000-48000 Hz | Must match microphone sample rate |
| `window_size` | 100 | 10-500 ms | Analysis window duration |
| `energy_threshold` | 100.0 | 0+ | Goertzel energy threshold (tune this) |
| `rms_threshold` | 0.0069 | 0-1.0 | RMS amplitude threshold |
| `min_duration` | 40 | 10-1000 ms | Minimum valid beep duration |
| `max_duration` | 100 | 10-1000 ms | Maximum valid beep duration |
| `cooldown` | 200 | 0-5000 ms | Cooldown after detection |
| `debounce_count` | 2 | 1-10 | Consecutive detections required |

### Exposed Sensors

1. **Binary Sensor**: "Beep Detected"
   - State: ON when beep is detected, OFF otherwise
   - Device class: sound
   - 100ms delayed_off filter for visibility

2. **Energy Sensor**: "Beep Energy Level"
   - Real-time Goertzel energy at target frequency
   - Use for threshold calibration

3. **RMS Sensor**: "Audio RMS Level"
   - Overall audio amplitude
   - Use for ambient noise assessment

4. **Count Sensor**: "Total Beep Detections"
   - Cumulative count of confirmed detections
   - Resets on device reboot

## Building and Deployment

### Prerequisites

1. ESPHome installed and configured
2. M5Stack Atom Echo connected via USB
3. WiFi credentials in `secrets.yaml`

### Build and Flash

```bash
# Navigate to project directory
cd /Users/sam/tmp/esphome-audio-sensor

# Validate configuration
esphome config esphome-atom-d4d5d0.yaml

# Build and upload
esphome run esphome-atom-d4d5d0.yaml

# Or use docker
docker run --rm -v "${PWD}":/config -it esphome/esphome run esphome-atom-d4d5d0.yaml
```

### Monitor Logs

```bash
# View real-time logs
esphome logs esphome-atom-d4d5d0.yaml

# Or via serial
esphome logs esphome-atom-d4d5d0.yaml --device /dev/cu.usbserial-*
```

## Calibration Guide

### Step 1: Baseline Assessment

1. Flash the device with default configuration
2. Monitor the energy and RMS sensors in Home Assistant
3. Observe baseline noise levels (no beep present)
4. Note the typical background energy/RMS values

### Step 2: Beep Exposure

1. Play your beep sound near the device
2. Watch the energy sensor spike
3. Note the peak energy value during beep
4. Check that RMS also increases

### Step 3: Threshold Tuning

**If detecting too many false positives:**
- Increase `energy_threshold` (try 150.0, 200.0)
- Increase `rms_threshold` (try 0.010, 0.015)
- Increase `debounce_count` (try 3, 4)

**If missing real beeps:**
- Decrease `energy_threshold` (try 75.0, 50.0)
- Decrease `rms_threshold` (try 0.005, 0.003)
- Decrease `debounce_count` (try 1)
- Verify `target_frequency` matches your beep

**If detecting beeps that are too long/short:**
- Adjust `min_duration` / `max_duration` based on actual beep timing
- Check logs for "Detection failed duration check" messages

### Step 4: Fine-Tuning

Example modified YAML configuration:

```yaml
beep_detector:
  id: beep_detector_component
  microphone: echo_microphone

  # Tuned parameters (example)
  target_frequency: 2615.0
  energy_threshold: 150.0    # Increased from 100.0
  rms_threshold: 0.010       # Increased from 0.0069
  debounce_count: 3          # Increased from 2

  # Rest of config...
```

## Monitoring and Debugging

### Key Log Messages

```
[D][beep_detector:XXX] Beep detected: energy=234.56 (thresh=100.00), rms=0.0123 (thresh=0.0069)
[I][beep_detector:XXX] Beep CONFIRMED! Duration: 65 ms, Total: 42
[D][beep_detector:XXX] State: IDLE -> DETECTING
[D][beep_detector:XXX] Detection failed duration check: 150 ms (range: 40-100 ms)
```

### Home Assistant Integration

The following entities will appear in Home Assistant:

- **binary_sensor.beep_detected**: Main detection state
- **sensor.beep_energy_level**: Real-time frequency energy
- **sensor.audio_rms_level**: Real-time audio amplitude
- **sensor.total_beep_detections**: Cumulative count

Create automations in Home Assistant:

```yaml
automation:
  - alias: "Beep Alert"
    trigger:
      - platform: state
        entity_id: binary_sensor.beep_detected
        to: 'on'
    action:
      - service: notify.mobile_app
        data:
          message: "Beep detected in kitchen!"
```

## Performance Characteristics

### Resource Usage

- **RAM**: ~3-5 KB for audio buffers and state
- **CPU**: Minimal (<5% on ESP32)
- **Update Rate**: Every 50ms (configurable)
- **Latency**: ~100-200ms from beep start to detection

### Optimization Notes

- Goertzel is significantly faster than full FFT
- Only processes target frequency (no wideband analysis)
- Sliding window with 25% overlap for smooth detection
- State machine prevents duplicate detections

## Troubleshooting

### No Audio Data

**Symptom**: Energy and RMS sensors always show 0

**Solutions**:
- Verify I2S pin configuration matches M5Stack Atom Echo
- Check microphone is enabled: `pdm: true`
- Verify sample rate matches: `sample_rate: 16000`
- Check logs for I2S initialization errors

### False Positives

**Symptom**: Detects beeps when none are present

**Solutions**:
- Increase `energy_threshold`
- Increase `rms_threshold`
- Increase `debounce_count`
- Adjust `target_frequency` if your beep is different
- Check for environmental noise sources

### Missing Detections

**Symptom**: Real beeps not detected

**Solutions**:
- Decrease thresholds
- Verify beep frequency matches `target_frequency`
- Check microphone positioning and distance
- Increase microphone gain (if available)
- Check beep duration is within min/max range

### Build Errors

**Symptom**: Compilation failures

**Solutions**:
- Ensure ESPHome version >= 2025.9.0
- Verify ESP-IDF framework is being used
- Check all files are in correct directories
- Clean build: `esphome clean esphome-atom-d4d5d0.yaml`

## Advanced Customization

### Multiple Frequency Detection

To detect multiple beep frequencies, duplicate the component:

```yaml
beep_detector:
  - id: beep_detector_2600
    microphone: echo_microphone
    target_frequency: 2600.0
    binary_sensor:
      name: "Beep 2600Hz Detected"

  - id: beep_detector_3000
    microphone: echo_microphone
    target_frequency: 3000.0
    binary_sensor:
      name: "Beep 3000Hz Detected"
```

### Custom Frequency Range

For frequency sweeps, use a bandpass approach:

```yaml
beep_detector:
  target_frequency: 2615.0  # Center frequency
  energy_threshold: 80.0    # Lower threshold
  # Monitor multiple instances at different frequencies
```

### Integration with Voice Assistant

The M5Stack Atom Echo can also run ESPHome voice assistant alongside beep detection:

```yaml
# Add voice assistant (if desired)
voice_assistant:
  microphone: echo_microphone
  # ... voice assistant config

# Beep detector runs in parallel
beep_detector:
  microphone: echo_microphone
  # ... beep detector config
```

**Note**: Sharing the microphone between components requires careful resource management.

## Algorithm Details

### Goertzel Algorithm

The Goertzel algorithm efficiently computes a single DFT bin (frequency component):

```
For target frequency f and sample rate fs:
  k = round(N * f / fs)
  ω = 2π * k / N
  coeff = 2 * cos(ω)

For each sample x[n]:
  q0 = coeff * q1 - q2 + x[n]
  q2 = q1
  q1 = q0

Magnitude² = q1² + q2² - q1*q2*coeff
```

This is O(N) complexity vs O(N log N) for FFT, with minimal memory overhead.

### State Machine

```
IDLE: Waiting for detection
  ↓ (energy > threshold AND rms > threshold)
DETECTING: Accumulating consecutive detections
  ↓ (debounce_count reached AND duration valid)
CONFIRMED: Beep validated
  ↓ (immediately)
COOLDOWN: Prevent duplicate detections
  ↓ (after cooldown_ms)
IDLE: Ready for next detection
```

## References

- [ESPHome Custom Components](https://esphome.io/custom/custom_component.html)
- [M5Stack Atom Echo Documentation](https://docs.m5stack.com/en/core/atom_echo)
- [Goertzel Algorithm](https://en.wikipedia.org/wiki/Goertzel_algorithm)
- [I2S Audio in ESPHome](https://esphome.io/components/i2s_audio.html)

## License

This implementation is provided as-is for use with ESPHome and Home Assistant.

## Support

For issues and questions:
1. Check ESPHome logs for error messages
2. Verify hardware connections and configuration
3. Test with known beep sound source
4. Calibrate thresholds using diagnostic sensors
