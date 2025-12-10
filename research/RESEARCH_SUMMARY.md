# Research Summary: ESPHome Audio Sensor for Beep Detection

**Research Completed**: 2025-12-10
**Target Platform**: M5Stack Atom Echo (ESP32-PICO-D4)
**Objective**: Real-time appliance beep detection with Home Assistant integration
**Status**: ✅ FEASIBLE - Ready for implementation

---

## Executive Summary

The M5Stack Atom Echo is **well-suited for beep detection** with ESPHome. Research confirms that both simple FFT-based and ML-based approaches fit within hardware constraints. The recommended path is to start with FFT-based frequency detection, then upgrade to machine learning only if needed for complex scenarios.

### Key Findings

✅ **Hardware Capable**: ESP32-PICO-D4 has sufficient memory (520 KB RAM, 4 MB flash)
✅ **ESPHome Support**: Native I2S/PDM microphone support with proven configurations
✅ **Multiple Approaches**: FFT (simple) and TensorFlow Lite (robust) both viable
✅ **Active Ecosystem**: Strong community support, libraries, and examples
✅ **Low Latency**: 50-250ms detection latency (depending on approach)

⚠️ **Key Constraint**: Must use PDM mode for SPM1423 microphone
⚠️ **Resource Conflict**: Cannot run Bluetooth + audio + ML simultaneously

---

## Documentation Structure

This research package contains four documents:

1. **TECHNICAL_BRIEF.md** (this summary + deep technical details)
   - Comprehensive hardware specifications
   - Audio preprocessing theory
   - ML model constraints and options
   - Memory budgets and performance analysis
   - Implementation approaches comparison
   - Risk assessment

2. **QUICK_REFERENCE.md** (implementation cheat sheet)
   - Hardware quick facts
   - ESPHome configuration snippets
   - Memory budget table
   - Decision tree for approach selection
   - Testing checklist
   - Common pitfalls

3. **IMPLEMENTATION_EXAMPLES.md** (code examples)
   - Complete YAML configurations
   - C++ external component examples (FFT-based)
   - Edge Impulse integration template
   - Debugging utilities
   - Home Assistant automation examples
   - Performance optimization tips

4. **RESEARCH_SUMMARY.md** (this file)
   - Executive overview
   - Recommendation matrix
   - Quick start guide
   - Key references

---

## Recommended Implementation Strategy

### Phase 1: FFT-Based Detection (START HERE)

**Why**: Fastest path to working prototype, no training data needed

**Timeline**: 1-2 days
**Complexity**: Medium
**Risk**: Low

**Steps**:
1. Configure M5Stack Atom Echo with ESPHome (I2S + PDM)
2. Verify audio capture using existing voice_assistant component
3. Create custom external component with FFT library (SoundAnalyzer)
4. Implement frequency band energy detection (2-4 kHz typical)
5. Tune thresholds with actual appliance beeps
6. Deploy and test for 24+ hours

**Expected Results**:
- Detection latency: 50-100ms
- RAM usage: ~150 KB
- Accuracy: >90% (with proper tuning)
- Works for beeps at known frequencies

**When This is Sufficient**:
- Appliance beeps at consistent frequency (e.g., 2500 Hz tone)
- Minimal background noise
- Single beep type to detect
- Speed priority over flexibility

### Phase 2: ML-Based Detection (IF NEEDED)

**Why**: More robust, handles complex beep patterns, better noise rejection

**Timeline**: 3-5 days (including training)
**Complexity**: High
**Risk**: Medium

**Triggers for Upgrade**:
- Multiple different beep types to distinguish
- Variable frequency beeps (chirps, sweeps)
- High background noise environment
- FFT approach has too many false positives/negatives

**Steps**:
1. Collect training data (10+ min per class: beeps, background)
2. Upload to Edge Impulse platform
3. Configure impulse (1000ms window, MFE preprocessing, CNN classifier)
4. Train model and validate (target >90% accuracy)
5. Export as Arduino library (INT8 quantized)
6. Integrate into ESPHome external component
7. Deploy and validate

**Expected Results**:
- Detection latency: 150-250ms
- RAM usage: ~300 KB
- Accuracy: >95% (with good training data)
- Robust to noise and variations

---

## Hardware Configuration Matrix

| Component | Specification | Notes |
|-----------|---------------|-------|
| **SoC** | ESP-PICO-D4 (ESP32 @ 240 MHz) | Dual-core Xtensa LX6 |
| **RAM** | 520 KB SRAM | ~250-350 KB available after OS |
| **Flash** | 4 MB | ~2.5 MB available for app/model |
| **Microphone** | SPM1423 (PDM MEMS) | **Must enable PDM mode** |
| **I2S Pins** | G19 (BCLK), G33 (LRCLK), G23 (DIN) | **Do not reuse these pins** |
| **Sample Rate** | 16 kHz (recommended) | 8 kHz for lower CPU usage |
| **Framework** | ESP-IDF (not Arduino) | Required for ESPHome audio |

---

## Memory Budget Comparison

### FFT-Based Approach

| Component | RAM | Flash |
|-----------|-----|-------|
| ESPHome + I2S | ~100 KB | ~850 KB |
| Audio Buffer (1s) | ~32 KB | - |
| FFT Library | ~30 KB | ~50 KB |
| **TOTAL** | **~162 KB** | **~900 KB** |
| **AVAILABLE** | **520 KB** | **4096 KB** |
| **MARGIN** | **358 KB** ✅ | **3196 KB** ✅ |

### ML-Based Approach (TFLite + Edge Impulse)

| Component | RAM | Flash |
|-----------|-----|-------|
| ESPHome + I2S | ~100 KB | ~850 KB |
| Audio Buffer (1s) | ~32 KB | - |
| Preprocessing | ~30 KB | ~50 KB |
| TFLite Runtime | ~50 KB | ~100 KB |
| Model + Tensor Arena | ~120 KB | ~300 KB |
| **TOTAL** | **~332 KB** | **~1300 KB** |
| **AVAILABLE** | **520 KB** | **4096 KB** |
| **MARGIN** | **188 KB** ✅ | **2796 KB** ✅ |

**Conclusion**: Both approaches fit comfortably within ESP32 constraints.

---

## Performance Expectations

### FFT-Based Detection

| Metric | Value | Notes |
|--------|-------|-------|
| Latency | 50-100ms | Audio capture + FFT + detection |
| CPU Usage | ~50% @ 240 MHz | Can reduce clock to 160 MHz |
| Accuracy | >90% | Depends on threshold tuning |
| False Positives | Low-Medium | Threshold-dependent |
| Training Required | None | Configure frequency range only |
| Noise Robustness | Medium | Struggles with loud background |

### ML-Based Detection (Edge Impulse)

| Metric | Value | Notes |
|--------|-------|-------|
| Latency | 150-250ms | Preprocessing + inference |
| CPU Usage | ~70% @ 240 MHz | Recommend full 240 MHz |
| Accuracy | >95% | With good training data |
| False Positives | Very Low | Learned discrimination |
| Training Required | 10-20 min/class | One-time data collection |
| Noise Robustness | High | Trained on diverse samples |

---

## Critical Configuration Requirements

### ESPHome YAML (Minimal Working Config)

```yaml
esp32:
  board: m5stack-atom
  framework:
    type: esp-idf  # NOT arduino

i2s_audio:
  i2s_lrclk_pin: GPIO33
  i2s_bclk_pin: GPIO19

microphone:
  - platform: i2s_audio
    id: atom_mic
    i2s_din_pin: GPIO23
    adc_type: external
    pdm: true  # CRITICAL FOR SPM1423
    sample_rate: 16000
    bits_per_sample: 32bit
```

### Common Mistakes to Avoid

❌ Using Arduino framework (lacks audio features)
❌ Forgetting `pdm: true` (microphone won't work)
❌ Enabling Bluetooth with audio (resource conflict/crashes)
❌ Reusing I2S pins (hardware damage risk)
❌ Dynamic memory allocation in audio callbacks (crashes)
❌ Starting with ML before testing FFT approach

---

## Key Libraries & Tools

### For FFT-Based Approach

1. **SoundAnalyzer** (RECOMMENDED)
   - URL: https://github.com/MichielFromNL/SoundAnalyzer
   - Features: FFT, MFCC, RMS, dBSPL
   - Performance: ~20ms for 1024 samples
   - Memory: Efficient, static allocation

2. **ESP-DSP** (Espressif Official)
   - URL: https://github.com/espressif/esp-dsp
   - Features: Optimized FFT, filters, matrix ops
   - Performance: Assembly-optimized
   - Integration: ESP-IDF native

### For ML-Based Approach

1. **Edge Impulse** (RECOMMENDED)
   - URL: https://edgeimpulse.com
   - Features: Complete ML pipeline (web-based)
   - Export: Arduino library, C++ library
   - Models: INT8 quantized, ESP32-optimized
   - Free tier: Sufficient for this project

2. **TensorFlow Lite Micro** (Manual)
   - URL: https://github.com/tensorflow/tflite-micro
   - Features: Lightweight ML runtime
   - Use Case: Custom models, full control
   - Complexity: High (manual integration)

### For ESPHome Integration

1. **ESPHome Audio Components** (ADF Wrapper)
   - URL: https://github.com/gnumpi/esphome_audio
   - Features: Espressif ADF access in ESPHome
   - Use Case: Advanced audio processing

2. **Official ESPHome Examples**
   - Atom Echo Config: https://github.com/esphome/wake-word-voice-assistants

---

## Testing & Validation Checklist

### Hardware Validation
- [ ] Microphone captures audio (test with voice_assistant component)
- [ ] No static/clicking sounds (verify PDM config)
- [ ] Audio level appropriate (not too quiet/loud)
- [ ] LED responds to commands (verify GPIO control)
- [ ] Wi-Fi stable (check RSSI in logs)

### FFT Implementation
- [ ] FFT computes without crashes (check heap usage)
- [ ] Frequency bins calculated correctly (log and verify)
- [ ] Target frequency range contains beep energy (debug spectrum)
- [ ] Background noise properly rejected (test in quiet/noisy)
- [ ] Threshold tuned for actual appliance (minimize false pos/neg)

### ML Implementation (if used)
- [ ] Training data balanced (equal time per class)
- [ ] Model accuracy >90% on test set (Edge Impulse dashboard)
- [ ] INT8 quantization applied (check model export)
- [ ] Inference completes in <500ms (measure on device)
- [ ] Memory usage stable (no leaks over 24 hours)

### Integration
- [ ] Home Assistant discovers binary sensor
- [ ] State updates within 1 second of beep
- [ ] No false triggers from other sounds
- [ ] Confidence sensor updates (if implemented)
- [ ] Automation triggers correctly

### Stability
- [ ] Runs for 24+ hours without crash
- [ ] Free heap stays above 100 KB
- [ ] CPU temperature stable (ESP32 has no temp sensor, but check for warm spots)
- [ ] Wi-Fi reconnects after router reboot
- [ ] OTA updates work (for future firmware)

---

## Quick Start Guide (30-Minute Test)

### Step 1: Install ESPHome (5 min)
```bash
pip install esphome
esphome wizard atom-echo-test.yaml
```

### Step 2: Configure M5Stack Atom Echo (10 min)
Copy the basic configuration from QUICK_REFERENCE.md, add your Wi-Fi credentials

### Step 3: Flash Device (5 min)
```bash
esphome run atom-echo-test.yaml
# Select USB serial port
```

### Step 4: Verify Audio (5 min)
Check logs for microphone initialization:
```
[I][i2s_audio:123]: I2S Audio initialized
[I][microphone:456]: Microphone started
```

### Step 5: Test Integration (5 min)
Add to Home Assistant, verify device appears and reports state

**Success Criteria**: Device online, microphone initializes, no crash after 5 minutes

---

## Recommended Next Steps for Implementation

### Immediate (Next 1-2 Days)
1. Set up M5Stack Atom Echo with basic ESPHome config
2. Verify audio capture works (use existing voice_assistant example)
3. Record sample audio from target appliance
4. Analyze frequency spectrum to determine beep characteristics

### Short-Term (Next 3-5 Days)
1. Implement FFT-based beep detector (custom component)
2. Tune frequency range and threshold for target appliance
3. Test in realistic conditions (various noise levels)
4. Validate 24-hour stability

### Medium-Term (If FFT Insufficient)
1. Collect training data (beeps + background noise)
2. Train Edge Impulse model
3. Integrate TFLite model into ESPHome
4. Validate accuracy improvement vs FFT approach

### Long-Term (Optimization & Deployment)
1. Reduce power consumption (CPU frequency scaling)
2. Add configuration UI (ESPHome web interface)
3. Implement advanced features (beep counting, pattern recognition)
4. Document for community sharing (ESPHome forum, GitHub)

---

## Support Resources

### Documentation
- **ESPHome Docs**: https://esphome.io/components/i2s_audio/
- **M5Stack Docs**: https://docs.m5stack.com/en/atom/atomecho
- **ESP32 Audio Guide**: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html

### Community
- **ESPHome Discord**: https://discord.gg/KhAMKrd
- **Home Assistant Forum**: https://community.home-assistant.io/c/esphome/32
- **Edge Impulse Forum**: https://forum.edgeimpulse.com
- **M5Stack Community**: https://community.m5stack.com

### GitHub Repositories
- **ESPHome**: https://github.com/esphome/esphome
- **ESPHome Issues**: https://github.com/esphome/issues
- **Example Configs**: https://github.com/esphome/wake-word-voice-assistants
- **SoundAnalyzer**: https://github.com/MichielFromNL/SoundAnalyzer

### Reference Projects
- ESP32 Audio Classification: https://forum.edgeimpulse.com/t/esp-32-audio-classification-inmp441/3439
- Offline Voice Assistant: https://www.hackster.io/ElectroScopeArchive/offline-esp32-voice-recognition-with-edge-impulse-fc93c9
- Sound Analyzer: https://www.instructables.com/Portable-Sound-Analyzer-on-ESP32/

---

## Risk Assessment Summary

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Memory Overflow** | Medium | High | Start with FFT approach, profile carefully, use static buffers |
| **CPU Overload** | Low | Medium | Use 8 kHz sample rate if needed, optimize algorithms |
| **PDM Config Issues** | Medium | High | Follow proven M5Stack Atom Echo configs exactly |
| **Poor Accuracy** | Medium | Medium | Collect diverse training data, tune thresholds systematically |
| **Integration Complexity** | Low | Medium | Use ESPHome external_components, follow examples |
| **Power Consumption** | Low | Low | Optimize CPU frequency if battery-powered |
| **Hardware Damage** | Very Low | Very High | Never reuse I2S pins (G19/G22/G23/G33) |

**Overall Risk**: LOW-MEDIUM (manageable with proper approach)

---

## Success Metrics

### Technical Metrics
- [ ] Detection accuracy: >90% (FFT) or >95% (ML)
- [ ] Detection latency: <500ms
- [ ] False positive rate: <5% over 24 hours
- [ ] False negative rate: <5% over 24 hours
- [ ] Stability: No crashes for 7+ days
- [ ] Memory: >100 KB free heap maintained
- [ ] CPU: <80% average utilization

### User Experience Metrics
- [ ] Home Assistant integration seamless
- [ ] Notifications arrive within 2 seconds of beep
- [ ] Zero missed critical alerts (e.g., washing machine done)
- [ ] Configuration via YAML (no code changes needed)
- [ ] OTA updates work reliably

---

## Conclusion

The M5Stack Atom Echo with ESPHome is **highly suitable** for appliance beep detection. The hardware has sufficient resources, ESPHome provides excellent audio support, and multiple implementation approaches are available.

**Recommended Path**:
1. Start with **FFT-based detection** for fastest results
2. Upgrade to **Edge Impulse ML** if more robustness needed
3. Follow the **implementation examples** provided
4. Use the **quick reference** for configuration
5. Leverage the **active community** for support

**Estimated Time to Working Prototype**:
- FFT Approach: 1-2 days
- ML Approach: 3-5 days (including training)

**Expected Accuracy**:
- FFT: >90% with proper tuning
- ML: >95% with good training data

**Status**: ✅ Ready for implementation

---

**Research Completed By**: Hardware & Audio Research Specialist
**Date**: 2025-12-10
**Project**: ESPHome Audio Sensor for Beep Detection
**Documents**: 4 files (Technical Brief, Quick Reference, Implementation Examples, Research Summary)
**Status**: COMPLETE & IMPLEMENTATION-READY
