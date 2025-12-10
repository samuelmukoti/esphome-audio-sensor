# Quick Reference: M5Stack Atom Echo Beep Detection

## Hardware Quick Facts

**M5Stack Atom Echo**
- ESP-PICO-D4: ESP32 dual-core @ 240 MHz
- RAM: 520 KB | Flash: 4 MB
- Microphone: SPM1423 (PDM mode required)
- Pins: G23 (data), G22/G19/G33 (I2S) - DO NOT REUSE

## ESPHome Configuration Snippet

```yaml
# I2S Audio Bus
i2s_audio:
  i2s_lrclk_pin: GPIO33
  i2s_bclk_pin: GPIO19

# PDM Microphone
microphone:
  - platform: i2s_audio
    id: atom_mic
    i2s_din_pin: GPIO23
    adc_type: external
    pdm: true              # CRITICAL for SPM1423
    sample_rate: 16000     # 16kHz standard
    bits_per_sample: 32bit
    channel: right
```

## Memory Budget (ML Approach)

| Component | RAM | Flash |
|-----------|-----|-------|
| ESPHome + I2S | ~100 KB | ~850 KB |
| Audio Buffer | ~32 KB | - |
| Preprocessing | ~30 KB | ~50 KB |
| TFLite + Model | ~170 KB | ~400 KB |
| **TOTAL** | **~332 KB** | **~1.3 MB** |
| **MARGIN** | **188 KB** ✅ | **2.7 MB** ✅ |

## Recommended Libraries

**FFT/Audio Processing:**
- SoundAnalyzer: https://github.com/MichielFromNL/SoundAnalyzer
- ESP-DSP: https://github.com/espressif/esp-dsp

**Machine Learning:**
- Edge Impulse: https://docs.edgeimpulse.com
- TFLite Micro: Built into ESP-IDF

## Implementation Decision Tree

```
Start → Need to detect beeps at KNOWN frequency?
         ├─ YES → Use FFT approach (Approach A)
         │        - Faster (50ms latency)
         │        - Less memory (~150 KB RAM)
         │        - No training needed
         │        - Library: SoundAnalyzer
         │
         └─ NO → Need to classify MULTIPLE beep types?
                  ├─ YES → Use Edge Impulse ML (Approach B)
                  │        - More robust (noise rejection)
                  │        - Slower (200ms latency)
                  │        - More memory (~300 KB RAM)
                  │        - Requires training data (10 min/class)
                  │
                  └─ NO → Start with FFT, upgrade later if needed
```

## Performance Targets

**FFT-Based Detection:**
- Latency: 50-100ms
- RAM: ~150 KB
- Accuracy: >90% (with tuned thresholds)
- CPU: ~50% @ 240 MHz

**ML-Based Detection:**
- Latency: 150-250ms
- RAM: ~300 KB
- Accuracy: >95% (with good training data)
- CPU: ~70% @ 240 MHz

## Critical Configuration Notes

1. **PDM Mode**: MUST enable `pdm: true` for SPM1423
2. **Sample Rate**: Start with 16kHz, try 8kHz if CPU-bound
3. **Pin Protection**: Never reuse G19/G22/G23/G33
4. **Bluetooth**: Disable BLE when using audio (resource conflict)
5. **Framework**: Use `esp-idf` (not Arduino) for audio features

## Testing Checklist

- [ ] Verify I2S microphone captures audio (test with voice_assistant)
- [ ] Confirm no static/noise (check DC offset correction)
- [ ] Record audio sample and analyze spectrum (verify frequency range)
- [ ] Test with actual appliance beep (not phone/computer tone)
- [ ] Validate in noisy environment (TV, conversation, etc.)
- [ ] Check memory usage: `ESP.getFreeHeap()` should show >100KB free
- [ ] Run 24-hour stability test

## Common Pitfalls

❌ **Don't**: Use Arduino framework (missing audio features)
✅ **Do**: Use esp-idf framework in ESPHome

❌ **Don't**: Forget PDM mode (SPM1423 won't work)
✅ **Do**: Set `pdm: true` in microphone config

❌ **Don't**: Enable Bluetooth with audio (crashes)
✅ **Do**: Disable BLE when using audio processing

❌ **Don't**: Use dynamic memory in audio callbacks
✅ **Do**: Pre-allocate buffers statically

❌ **Don't**: Start with ML if beep frequency is known
✅ **Do**: Start with FFT, upgrade to ML only if needed

## Key Links

**Essential Documentation:**
- ESPHome I2S: https://esphome.io/components/i2s_audio/
- Atom Echo Example: https://github.com/esphome/wake-word-voice-assistants/blob/main/m5stack-atom-echo/m5stack-atom-echo.yaml

**Implementation Examples:**
- SoundAnalyzer: https://github.com/MichielFromNL/SoundAnalyzer (FFT)
- Edge Impulse Audio: https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/audio/audio-classification (ML)

**Community Help:**
- ESPHome Discord: https://discord.gg/KhAMKrd
- Edge Impulse Forum: https://forum.edgeimpulse.com

## Quick Start Command Sequence

```bash
# 1. Create ESPHome project
esphome wizard atom-echo-beep.yaml

# 2. Add I2S configuration (see snippet above)

# 3. Install SoundAnalyzer library (for FFT approach)
# Add to platformio_options in yaml:
#   lib_deps:
#     - https://github.com/MichielFromNL/SoundAnalyzer

# 4. Create external component (custom C++)
# See: https://esphome.io/components/external_components.html

# 5. Flash to device
esphome run atom-echo-beep.yaml
```

## Support Contacts

- **Hardware Issues**: M5Stack forum / GitHub issues
- **ESPHome Integration**: ESPHome Discord / GitHub discussions
- **ML Training**: Edge Impulse forum / community
- **ESP32 Audio**: Espressif forum / esp-sr GitHub

---

**Last Updated**: 2025-12-10
**Status**: Ready for Implementation
