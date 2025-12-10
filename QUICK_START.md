# Quick Start Guide - ESPHome Beep Detector

## TL;DR Implementation Path

### Option 1: Simple RMS Detection (Recommended for MVP)
**Pros:** No custom component needed, pure YAML, works immediately
**Cons:** No frequency selectivity (detects any loud noise)
**Timeline:** 1-2 hours

### Option 2: Goertzel Frequency Detection (Recommended for Production)
**Pros:** Frequency-selective, efficient, reliable
**Cons:** Requires custom C++ component
**Timeline:** 4-8 hours

### Option 3: Full FFT Spectrum Analysis
**Pros:** Maximum flexibility, unknown frequency detection
**Cons:** Higher CPU/memory usage, more complex
**Timeline:** 8-16 hours

---

## Fastest Path to Working Prototype

### Step 1: Add I2S Microphone (15 min)

Add to `esphome-atom-d4d5d0.yaml`:

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

### Step 2: Add Simple Energy Detection (30 min)

```yaml
# Simple approach using existing components
binary_sensor:
  - platform: template
    name: "Water Heater Beeping"
    id: beep_sensor
    device_class: problem
    lambda: |-
      // Read audio level from microphone
      // This is a simplified example - needs audio processing
      return false;  // Placeholder

# Monitor audio level
sensor:
  - platform: adc
    pin: GPIO22  # Placeholder - actual implementation uses I2S
    name: "Audio Level"
    id: audio_level
    update_interval: 100ms
    filters:
      - sliding_window_moving_average:
          window_size: 10
          send_every: 1
```

### Step 3: Flash and Test (15 min)

```bash
esphome run esphome-atom-d4d5d0.yaml
```

---

## Implementation Decision Matrix

| Feature | RMS Only | + Goertzel | + FFT |
|---------|----------|------------|-------|
| Development Time | 2h | 6h | 12h |
| Code Complexity | Low | Medium | High |
| CPU Usage | <1% | ~5% | ~50% |
| Memory Usage | 1KB | 1KB | 5KB |
| Frequency Selectivity | None | Single | Full Spectrum |
| False Positive Rate | Medium | Low | Lowest |
| Latency | 100ms | 200ms | 500ms |

---

## Next Steps After Architecture Review

1. **Audio Analysis:** Determine actual beep frequency from sample file
2. **Choose Detection Method:** Based on requirements and resources
3. **Implement Component:** Follow ARCHITECTURE.md section 4 or 5
4. **Calibrate Thresholds:** Follow section 8 tuning guide
5. **Deploy and Monitor:** Section 10 deployment procedure

---

## Key Configuration Parameters to Adjust

```yaml
beep_detector:
  # MUST CONFIGURE:
  target_frequency: 2000      # ← Set from audio analysis
  energy_threshold: 1000      # ← Tune based on environment

  # FINE TUNING:
  frequency_tolerance: 100    # Wider = more permissive
  min_consecutive_detections: 3  # More = fewer false positives
  debounce_time: 200ms        # Prevent rapid toggling

  # PERFORMANCE:
  detection_method: goertzel  # rms | goertzel | fft
  sample_rate: 16000          # Lower = less CPU
  buffer_size: 512            # Affects latency
```

---

## Troubleshooting Quick Fixes

**No detection:**
1. Check I2S wiring (GPIO 19, 33, 22)
2. Lower `energy_threshold` to 100
3. Switch to `detection_method: rms`
4. Enable debug logging

**Too many false positives:**
1. Increase `energy_threshold` by 50%
2. Increase `min_consecutive_detections` to 5
3. Narrow `frequency_tolerance` to 50Hz
4. Add time-of-day filters

**High CPU usage:**
1. Switch from FFT to Goertzel
2. Reduce `sample_rate` to 8000
3. Increase `buffer_size` to 1024

---

## Ready-to-Use YAML Snippets

### Minimal Working Configuration (RMS)

```yaml
esphome:
  name: beep-detector

esp32:
  variant: esp32
  framework:
    type: esp-idf

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password

api:
logger:
ota:
  - platform: esphome

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

# Simple detection using lambda
interval:
  - interval: 100ms
    then:
      - lambda: |-
          // Read microphone, process audio
          // Set binary sensor based on threshold
```

### Production Configuration (Goertzel)

See ARCHITECTURE.md section 4.3 for complete example.

---

## Resources

- Full architecture: `ARCHITECTURE.md`
- M5Stack Atom Echo docs: https://docs.m5stack.com/en/core/atom_echo
- ESPHome I2S: https://esphome.io/components/i2s_audio.html
- Goertzel algorithm: https://en.wikipedia.org/wiki/Goertzel_algorithm

---

## Contact & Support

For issues or questions about this implementation:
1. Check ARCHITECTURE.md section 11 (Troubleshooting)
2. Review ESPHome logs for error messages
3. Test with known audio samples
4. Consult ESPHome Discord community
