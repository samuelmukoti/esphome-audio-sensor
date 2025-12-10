# Build and Deploy Instructions

## Quick Start

Your M5Stack Atom Echo beep detector is ready to deploy. Follow these steps:

### 1. Verify File Structure

```bash
cd /Users/sam/tmp/esphome-audio-sensor
ls -R
```

Expected structure:
```
.
├── esphome-atom-d4d5d0.yaml
├── custom_components/
│   └── beep_detector/
│       ├── __init__.py
│       ├── beep_detector.h
│       └── beep_detector.cpp
├── README_IMPLEMENTATION.md
└── BUILD_AND_DEPLOY.md
```

### 2. Validate Configuration

```bash
esphome config esphome-atom-d4d5d0.yaml
```

This will:
- Parse the YAML configuration
- Validate component registration
- Check for syntax errors
- Display component configuration

### 3. Build Firmware

```bash
esphome compile esphome-atom-d4d5d0.yaml
```

This will:
- Generate C++ code from YAML
- Compile custom components
- Build ESP-IDF firmware
- Create flashable binary

Expected output:
```
INFO Reading configuration esphome-atom-d4d5d0.yaml...
INFO Generating C++ source...
INFO Compiling app...
INFO Successfully compiled program.
```

### 4. Flash to Device

**Find your device:**
```bash
# macOS
ls /dev/cu.usbserial-*

# Linux
ls /dev/ttyUSB*
```

**Flash the firmware:**
```bash
# Option 1: Auto-detect and flash
esphome run esphome-atom-d4d5d0.yaml

# Option 2: Specify port explicitly
esphome upload esphome-atom-d4d5d0.yaml --device /dev/cu.usbserial-XXXXX
```

### 5. Monitor Logs

```bash
# Real-time log monitoring
esphome logs esphome-atom-d4d5d0.yaml

# Or specify device
esphome logs esphome-atom-d4d5d0.yaml --device /dev/cu.usbserial-XXXXX
```

## What to Look For in Logs

### Successful Boot

```
[C][beep_detector:XXX] Setting up Beep Detector...
[C][beep_detector:XXX]   Target Frequency: 2615.0 Hz
[C][beep_detector:XXX]   Sample Rate: 16000 Hz
[C][beep_detector:XXX]   Window Size: 100 ms (1600 samples)
[C][beep_detector:XXX]   Energy Threshold: 100.00
[C][beep_detector:XXX]   RMS Threshold: 0.0069
[C][beep_detector:XXX]   Duration Range: 40-100 ms
[C][beep_detector:XXX]   Goertzel Coefficient: 1.XXXXXX
```

### I2S Microphone Initialization

```
[C][i2s_audio:XXX] Setting up I2S Audio...
[D][i2s_audio:XXX] Starting I2S Audio Microphone
```

### Detection Events

```
[D][beep_detector:XXX] Beep detected: energy=234.56 (thresh=100.00), rms=0.0123 (thresh=0.0069)
[D][beep_detector:XXX] State: IDLE -> DETECTING
[I][beep_detector:XXX] Beep CONFIRMED! Duration: 65 ms, Total: 1
[D][beep_detector:XXX] State: CONFIRMED -> COOLDOWN
[D][beep_detector:XXX] State: COOLDOWN -> IDLE
```

## Initial Testing

### Step 1: Silent Environment Test

1. Flash and boot the device
2. Place in quiet environment
3. Monitor energy and RMS sensors
4. Record baseline values (should be low)

**Expected baseline:**
- Energy: 0-10
- RMS: 0.0001-0.001

### Step 2: Beep Test

1. Play your beep sound near the device (distance: 1-2 meters)
2. Watch logs for detection events
3. Check Home Assistant for "Beep Detected" binary sensor state change
4. Verify energy sensor spikes above threshold

**Expected during beep:**
- Energy: >100 (should exceed threshold)
- RMS: >0.0069 (should exceed threshold)
- Binary Sensor: ON
- Detection Count: Increments

### Step 3: Calibration

If detection doesn't work immediately:

**Too sensitive (false positives):**
```yaml
beep_detector:
  energy_threshold: 150.0  # Increase from 100.0
  rms_threshold: 0.010     # Increase from 0.0069
  debounce_count: 3        # Increase from 2
```

**Not sensitive enough (missing beeps):**
```yaml
beep_detector:
  energy_threshold: 50.0   # Decrease from 100.0
  rms_threshold: 0.005     # Decrease from 0.0069
  debounce_count: 1        # Decrease from 2
```

After changing values:
```bash
esphome run esphome-atom-d4d5d0.yaml  # Rebuild and flash
```

## Home Assistant Integration

### Add to Dashboard

```yaml
type: entities
title: Beep Detector
entities:
  - entity: binary_sensor.beep_detected
    name: Beep Status
  - entity: sensor.beep_energy_level
    name: Frequency Energy
  - entity: sensor.audio_rms_level
    name: Audio Level
  - entity: sensor.total_beep_detections
    name: Detection Count
```

### Create Automation

```yaml
automation:
  - alias: "Kitchen Beep Alert"
    description: "Alert when beep detected"
    trigger:
      - platform: state
        entity_id: binary_sensor.beep_detected
        to: 'on'
    action:
      - service: notify.mobile_app_your_phone
        data:
          title: "Beep Detected"
          message: "Device detected a beep at {{ now().strftime('%H:%M:%S') }}"
      - service: light.turn_on
        target:
          entity_id: light.kitchen_light
        data:
          flash: short
```

## Troubleshooting

### Build Fails

**Error: Component not found**
```
Solution: Ensure custom_components/beep_detector/ exists with all 3 files
```

**Error: Compilation failed**
```
Solution: Check ESP-IDF framework is specified in YAML
        Ensure ESPHome version >= 2025.9.0
```

### No Audio Data

**Symptom: Energy and RMS sensors always 0**
```
Solution: Check I2S pins match M5Stack Atom Echo hardware
        Verify PDM mode is enabled: pdm: true
        Check logs for I2S initialization errors
```

### Device Won't Connect to WiFi

**Symptom: Device boots but doesn't appear in Home Assistant**
```
Solution: Check secrets.yaml has correct WiFi credentials
        Verify WiFi network is 2.4GHz (ESP32 doesn't support 5GHz)
        Check device logs for WiFi error messages
```

### Detection Not Working

**Symptom: Device runs but never detects beeps**
```
Solution: Verify beep frequency matches target_frequency (2615 Hz)
        Check energy/RMS sensors are showing activity during beep
        Lower thresholds if sensors show activity below threshold
        Increase microphone volume or decrease distance to source
```

## Performance Monitoring

### Memory Usage

Check logs for heap usage:
```
[D][esp32:XXX] Free heap: XXXXX bytes
```

Should maintain >100KB free heap during operation.

### CPU Usage

The beep detector is designed for minimal CPU usage:
- Goertzel algorithm: O(N) complexity
- Update rate: 50ms intervals
- Expected CPU: <5% of ESP32 capacity

### Network Traffic

ESPHome will send sensor updates to Home Assistant:
- Binary sensor: On state change only
- Energy/RMS sensors: Every 50ms (configurable)
- Detection count: Every 50ms (configurable)

To reduce network traffic:
```yaml
beep_detector:
  # Add to component (requires code modification)
  # or use Home Assistant sensor filters
```

## Next Steps

1. **Optimize Thresholds**: Use diagnostic sensors to fine-tune detection
2. **Create Automations**: Set up Home Assistant automations for your use case
3. **Monitor Performance**: Track false positive/negative rates
4. **Adjust as Needed**: Modify parameters based on real-world performance

## Advanced Configuration

### Multiple Frequencies

Detect different beep types:
```yaml
# Not supported in current implementation
# Would require instantiating multiple detector instances
```

### Custom Analysis Window

Adjust processing window:
```yaml
beep_detector:
  window_size: 150  # Longer window for better frequency resolution
  # Trade-off: Slower detection response
```

### Production Settings

For 24/7 operation:
```yaml
logger:
  level: INFO  # Reduce log verbosity
  logs:
    beep_detector: INFO

beep_detector:
  cooldown: 500  # Longer cooldown to prevent rapid re-triggering
  debounce_count: 3  # More conservative detection
```

## Support Resources

- **ESPHome Docs**: https://esphome.io
- **M5Stack Atom Echo**: https://docs.m5stack.com/en/core/atom_echo
- **Home Assistant**: https://www.home-assistant.io

## Maintenance

### Firmware Updates

To update ESPHome version:
```bash
pip install -U esphome
esphome run esphome-atom-d4d5d0.yaml
```

### Backup Configuration

```bash
# Backup YAML and custom components
cp -r /Users/sam/tmp/esphome-audio-sensor ~/esphome-backup-$(date +%Y%m%d)
```

### Factory Reset

If needed, re-flash ESP32:
```bash
esphome run esphome-atom-d4d5d0.yaml --device /dev/cu.usbserial-XXXXX
# Select "Clean Build Files" option
```
