# Technical Brief: Beep Detection on M5Stack Atom Echo

**Research Date**: 2025-12-10
**Target Hardware**: M5Stack Atom Echo (ESP32-PICO-D4)
**Objective**: Implement real-time beep/tone detection for appliance monitoring

---

## 1. Hardware Specifications & Constraints

### 1.1 M5Stack Atom Echo Overview
- **Dimensions**: 24 x 24 x 17mm (ultra-compact)
- **SoC**: ESP-PICO-D4 (ESP32 dual-core @ 240 MHz)
- **Flash**: 4 MB SPI flash (integrated in ESP-PICO-D4)
- **RAM**: 520 KB SRAM (typical ESP32 internal memory)
  - 8 KB RTC FAST Memory (accessible during Deep-sleep)
  - 8 KB RTC SLOW Memory (co-processor accessible)
- **Connectivity**: 2.4 GHz Wi-Fi, Bluetooth/BLE
- **GPIO**: GROVE interface for expansion
- **LED**: SK6812 RGB LED for status indication

### 1.2 Audio Hardware

#### Microphone: SPM1423 (MEMS)
- **Type**: PDM (Pulse Density Modulation) microphone, NOT traditional I2S
- **Interface**: Digital output via PDM protocol
- **Pin Connections**:
  - MIC DATA → GPIO23
  - CLK → GPIO22
  - SYS LRCK → GPIO33
  - Additional I2S pin → GPIO19
- **Critical Note**: SPM1423 supports only clock and data lines (no frame select/word select), requiring PDM mode configuration in I2S driver

#### Speaker
- **Power**: 0.5W
- **Interface**: I2S DAC
- **Pins**:
  - DOUT → GPIO22
  - BCLK → GPIO19
  - LRCK → GPIO33

**⚠️ WARNING**: Pins G19/G22/G23/G33 are predefined and must not be reused to avoid hardware damage.

---

## 2. ESPHome Audio Capabilities

### 2.1 I²S Audio Microphone Component

ESPHome provides native I²S audio support for ESP32 with the following capabilities:

**Supported Configurations**:
- External ADC microphones via I2S bus
- Internal ESP32 ADC (standard ESP32 only, not variants)
- PDM microphones (ESP32 and ESP32-S3)

**Audio Parameters**:
- Sample rates: 8kHz - 48kHz (default: 16kHz)
- Bits per sample: 8/16/24/32-bit (default: 32-bit)
- Channels: mono (left/right) or stereo
- DC offset correction available
- APLL support for improved clock accuracy

**M5Stack Atom Echo Configuration Example**:
```yaml
i2s_audio:
  i2s_lrclk_pin: GPIO33
  i2s_bclk_pin: GPIO19

microphone:
  - platform: i2s_audio
    id: atom_mic
    i2s_din_pin: GPIO23
    adc_type: external
    pdm: true  # Critical for SPM1423
    sample_rate: 16000
    bits_per_sample: 32bit
    channel: right
```

### 2.2 Resource Constraints

**⚠️ CRITICAL**: ESPHome documentation warns that "Audio and voice components consume a significant amount of resources (RAM, CPU)" and "crashes are likely to occur" when combining too many components, especially Bluetooth/BLE features.

**Recommendations**:
- Disable Bluetooth when using audio processing
- Limit concurrent components
- Monitor memory usage carefully
- Use external components for advanced features

### 2.3 TensorFlow Lite Integration Status

**Current State**: ESPHome does NOT have built-in TensorFlow Lite Micro integration.

**Available Alternatives**:
1. **micro_wake_word** component (wake word detection only)
2. **voice_assistant** component (Home Assistant integration)
3. **Custom external components** (C++ integration with ESP-IDF/ADF)

**For ML-based beep detection**: Must use ESPHome external component system to integrate TFLite Micro or custom C++ libraries.

---

## 3. TensorFlow Lite Micro for Beep Detection

### 3.1 Model Size Constraints

**ESP32 Memory Budget**:
- Total SRAM: 520 KB
- Available for ML: ~100-200 KB (after system overhead)
- Flash for model: ~200-500 KB available

**Typical Model Sizes (INT8 quantized)**:
- Simple audio classification: 100-300 KB (flash)
- RAM requirements: 50-100 KB (tensor arena)
- Wake word detection: 100-300 KB (reference)

**✅ FEASIBLE**: Simple beep detection models fit comfortably within ESP32 constraints when using INT8 quantization.

### 3.2 Pre-trained Models & Platforms

#### Option A: Edge Impulse (RECOMMENDED)
- **Platform**: Cloud-based ML pipeline with ESP32 support
- **Training Requirements**: ~10 minutes of balanced audio data per class
- **Preprocessing**: MFE (Mel Frequency Energy) or MFCC
- **Output**: INT8 quantized TFLite model + C++ library
- **Deployment**: Arduino library or standalone C++ (ESP-IDF compatible)
- **Inference Speed**: ~175ms classification + ~400ms preprocessing
- **Advantages**:
  - Easy data collection via web/mobile
  - Automatic preprocessing pipeline
  - ESP32-optimized export
  - Active community & documentation
- **References**:
  - https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/audio/audio-classification
  - Multiple ESP32 + INMP441 audio projects on forum

#### Option B: Custom TFLite Models
- **Yamnet**: Too large for ESP32 (not recommended)
- **Micro Speech**: Google's example (keyword detection, not beep detection)
- **Custom CNN**: Train with TensorFlow, convert to TFLite, quantize to INT8
- **Advantages**: Full control over architecture
- **Disadvantages**: Requires ML expertise, manual optimization

#### Option C: ESP-SR (Espressif Speech Recognition)
- **Platform**: Espressif's official speech/audio framework
- **Features**: AFE (Audio Front-End), VAD, FFT, MFCC, Fbank
- **Models**: MultiNet (lightweight, ~20KB RAM, MFCC-based)
- **Use Case**: Primarily for speech, adaptable to sound classification
- **Repository**: https://github.com/espressif/esp-sr

### 3.3 Quantization Requirements

**INT8 Quantization is MANDATORY for ESP32**:
- Reduces model size by ~4x vs FP32
- 2x+ faster inference on ESP32-S3 (no FPU benefit on plain ESP32)
- Maintains >95% accuracy for most audio tasks
- All Edge Impulse models export as INT8 by default

**Quantization Process**:
1. Train model in FP32
2. Convert to TFLite format
3. Apply post-training quantization with representative dataset
4. Validate accuracy on test set

---

## 4. Audio Preprocessing Requirements

### 4.1 Recommended Pipeline

**For Beep Detection with ML**:
```
Raw Audio (PDM) → I2S Driver → Buffering → Preprocessing → ML Inference → Classification
                                  ↓
                            [FFT or MFCC]
```

**For Simple Beep Detection (Non-ML)**:
```
Raw Audio (PDM) → I2S Driver → FFT → Frequency Peak Detection → Threshold Comparison
```

### 4.2 Preprocessing Options

#### Option A: MFE (Mel Frequency Energy) - RECOMMENDED for ML
- **What**: Converts audio to frequency-time spectrogram
- **Advantages**:
  - Tuned to human auditory perception
  - Standard in Edge Impulse
  - Produces compact 2D representations
  - Good for varied beep frequencies
- **Parameters**:
  - Window size: 1000ms (typical)
  - Overlap: 300ms
  - Output: 2D feature matrix

#### Option B: MFCC (Mel Frequency Cepstral Coefficients)
- **What**: Audio features for speech/sound recognition
- **Advantages**:
  - Highly efficient (13-20 coefficients)
  - Excellent for ML models
  - Captures spectral envelope
- **Implementation**: Available in ESP-SR, SoundAnalyzer library
- **Performance**: ~20ms for 1024 samples @ 8192 Hz on ESP32

#### Option C: FFT (Fast Fourier Transform)
- **What**: Frequency domain analysis
- **Advantages**:
  - Simple and direct
  - Excellent for single-frequency beeps
  - Can bypass ML entirely
  - Lower computational cost
- **Implementation**: ESP-DSP, SoundAnalyzer library
- **Performance**: <20ms for 1024 samples on ESP32
- **Use Case**: Best for detecting beeps at known frequencies (e.g., 2000-4000 Hz)

### 4.3 Sample Rate & Windowing

**Recommended Settings**:
- **Sample Rate**: 16,000 Hz (standard for voice/sound)
  - Nyquist limit: 8,000 Hz (covers typical appliance beeps: 1-5 kHz)
  - 8,000 Hz option for lower CPU usage (if beeps <4 kHz)
- **Bit Depth**: 16-bit or 32-bit (ESPHome default: 32-bit)
- **Window Size**: 1024 samples (64ms @ 16kHz)
- **Overlap**: 50% (512 samples) for smoother detection
- **Buffer**: Circular buffer with 1-2 second capacity

### 4.4 Power Efficiency

**ESP32 Power Modes**:
- **Active (audio processing)**: ~160-240 mA @ 240 MHz
- **Light sleep**: Not practical (I2S requires active CPU)
- **Deep sleep**: Possible between detection events (if event-driven)

**Optimization Strategies**:
- Run audio at lower CPU frequency (80-160 MHz)
- Use dual-core: Core 0 for I2S, Core 1 for ML inference
- Batch processing: Buffer audio, process in chunks
- Dynamic frequency scaling: Reduce clock when idle
- Event-driven: Sleep between expected beep events

---

## 5. Performance Estimates

### 5.1 Memory Budget Analysis

| Component | Flash (KB) | RAM (KB) | Notes |
|-----------|------------|----------|-------|
| ESPHome Core | ~800 | ~50 | Base system |
| I2S Driver | ~50 | ~20 | Audio capture |
| Audio Buffer | 0 | ~32 | 1s @ 16kHz, 16-bit |
| Preprocessing | ~50 | ~30 | FFT/MFCC library |
| TFLite Micro | ~100 | ~50 | Runtime |
| ML Model | 150-300 | 70-100 | Tensor arena + model |
| **TOTAL** | **~1150-1350** | **~252-282** | |
| **AVAILABLE** | 4096 | 520 | ESP-PICO-D4 |
| **MARGIN** | **~2.7 MB** | **~240 KB** | ✅ FITS |

**✅ CONCLUSION**: TFLite-based beep detection is feasible within ESP32-PICO-D4 constraints.

### 5.2 Inference Performance

**Typical Timing (ESP32 @ 240 MHz)**:
- Audio capture: Continuous (I2S DMA)
- FFT (1024 samples): ~15-20 ms
- MFCC extraction: ~20 ms
- ML inference (simple CNN): ~100-200 ms
- **Total latency**: ~150-250 ms per classification

**Real-time Capability**:
- Window: 64ms (1024 samples @ 16kHz)
- Processing: 150-250ms
- **Result**: Can process ~4-6 classifications per second
- **Sufficient for beep detection**: Most appliance beeps last 200-1000ms

### 5.3 Accuracy Expectations

**With proper training data**:
- Beep vs background: >95% accuracy
- Multiple beep types: >90% accuracy
- Real-world robustness depends on:
  - Training data diversity
  - Background noise levels
  - Beep frequency stability

---

## 6. Implementation Approaches

### 6.1 APPROACH A: Simple FFT-Based Detection (NO ML)

**✅ RECOMMENDED FOR SIMPLE USE CASES**

**Concept**: Detect beeps at known frequencies using FFT + threshold

**Pros**:
- No training data required
- Fast (<50ms total latency)
- Lower memory usage (~150 KB RAM total)
- Easy to implement in ESPHome C++ component
- Deterministic behavior

**Cons**:
- Only works for beeps at known frequencies
- Less robust to background noise
- Cannot distinguish between different beep patterns

**Implementation**:
1. Use ESPHome I2S microphone component
2. Create custom C++ external component
3. Integrate SoundAnalyzer or ESP-DSP library for FFT
4. Analyze frequency bins for target range (e.g., 2-4 kHz)
5. Apply threshold + hysteresis
6. Publish binary sensor to Home Assistant

**Libraries**:
- **SoundAnalyzer**: https://github.com/MichielFromNL/SoundAnalyzer
  - Optimized for ESP32
  - FFT + MFCC support
  - ~20ms for 1024 samples
- **ESP-DSP**: Espressif official DSP library

**Example Detection Logic**:
```cpp
// Pseudo-code
float fft_magnitude[512];
compute_fft(audio_buffer, fft_magnitude);

// Check 2-4 kHz bins (assuming 16kHz sample rate)
float energy_2_4khz = sum(fft_magnitude[128:256]);
float total_energy = sum(fft_magnitude);

if (energy_2_4khz / total_energy > 0.6) {
    // Beep detected!
    publish_state(true);
}
```

### 6.2 APPROACH B: Edge Impulse ML Model

**✅ RECOMMENDED FOR COMPLEX/MULTI-CLASS DETECTION**

**Concept**: Train custom audio classifier via Edge Impulse platform

**Pros**:
- Handles multiple beep types
- Robust to background noise
- Easy training pipeline
- ESP32-optimized export
- Active community support

**Cons**:
- Requires training data collection (~10-20 min per class)
- Higher latency (~150-250ms)
- More memory usage (~250-300 KB RAM)
- Requires Edge Impulse account (free tier available)

**Implementation Steps**:
1. **Data Collection**:
   - Record 5-10 minutes of beep audio
   - Record 5-10 minutes of background/noise
   - Use phone or ESP32 itself
   - Upload to Edge Impulse

2. **Model Training**:
   - Configure impulse: 1000ms window, MFE preprocessing
   - Use default neural network (4 layers)
   - Train with data augmentation
   - Validate accuracy (target >90%)

3. **Model Export**:
   - Download as Arduino library (INT8 quantized)
   - OR download as standalone C++ library

4. **ESPHome Integration**:
   - Create external component
   - Include Edge Impulse library
   - Feed audio from I2S component to inference
   - Publish classification results to binary sensor

**Reference Projects**:
- ESP32 + INMP441 audio classification: https://forum.edgeimpulse.com/t/esp-32-audio-classification-inmp441/3439
- Faucet detection example: https://forum.edgeimpulse.com/t/audio-classification-using-esp32-and-inmp441-microphone-to-detect-faucet-on/17182

### 6.3 APPROACH C: Custom TFLite Model

**⚠️ NOT RECOMMENDED** (unless you have ML expertise)

**Concept**: Train custom model in TensorFlow, convert to TFLite, deploy manually

**Pros**:
- Full control over architecture
- Can optimize for specific use case
- No platform dependencies

**Cons**:
- Requires ML/DSP expertise
- Manual preprocessing pipeline
- Manual quantization and optimization
- Time-intensive development

**When to Use**: Only if Edge Impulse limitations are blocking (e.g., need for custom preprocessing, proprietary training data)

### 6.4 APPROACH D: ESP-SR Framework

**⚠️ PARTIAL RECOMMENDATION** (good for speech, overkill for beeps)

**Concept**: Use Espressif's speech recognition framework

**Pros**:
- Official Espressif support
- Comprehensive audio front-end (AFE, VAD, NS)
- Optimized for ESP32

**Cons**:
- Designed for speech, not generic sounds
- Larger framework (more complex integration)
- Less documentation for non-speech use cases

**When to Use**: If you need VAD (Voice Activity Detection) or other AFE features alongside beep detection

---

## 7. Recommended Implementation Path

### Phase 1: Proof of Concept (FFT-Based)
**Timeline**: 1-2 days
**Risk**: Low

1. Set up M5Stack Atom Echo with ESPHome
2. Configure I2S microphone in PDM mode
3. Create simple external component with SoundAnalyzer library
4. Implement FFT-based frequency detection
5. Test with actual appliance beeps
6. Tune frequency ranges and thresholds

**Deliverable**: Working beep detection for single frequency range

### Phase 2: ML Enhancement (if needed)
**Timeline**: 3-5 days
**Risk**: Medium

1. Collect training data (beeps + background)
2. Upload to Edge Impulse
3. Train and validate model
4. Export as Arduino library
5. Integrate into ESPHome external component
6. Test and compare to FFT approach

**Deliverable**: Robust multi-class beep classifier

### Phase 3: Optimization
**Timeline**: 1-2 days
**Risk**: Low

1. Profile memory usage
2. Optimize CPU frequency
3. Implement power-saving strategies
4. Add configuration options (sensitivity, frequency ranges)
5. Create comprehensive documentation

**Deliverable**: Production-ready component

---

## 8. Alternative Approaches (If ML is Too Resource-Intensive)

### 8.1 Goertzel Algorithm
**Concept**: Efficient single-frequency detection (lighter than full FFT)
- **Complexity**: O(N) vs FFT's O(N log N)
- **Use Case**: Detecting specific frequency (e.g., 2.5 kHz beep)
- **Memory**: Minimal (<10 KB)
- **Latency**: <10ms

### 8.2 Zero-Crossing Rate (ZCR)
**Concept**: Count audio signal zero-crossings to estimate frequency
- **Complexity**: Very low
- **Use Case**: Distinguishing beeps from speech/music
- **Limitations**: Not frequency-specific

### 8.3 Energy-Based Detection
**Concept**: Detect sudden increases in audio energy
- **Complexity**: Trivial
- **Use Case**: Generic "loud sound" detection
- **Limitations**: No frequency discrimination

### 8.4 Hardware Pre-Processing
**Concept**: Use analog bandpass filter (2-4 kHz) before ADC
- **Advantage**: Offload filtering from CPU
- **Disadvantage**: Requires hardware modification

---

## 9. Key Technical Recommendations

### 9.1 Start Simple
**Begin with FFT-based detection** (Approach A):
- No training data required
- Fast iteration and debugging
- Lower complexity
- Upgrade to ML only if needed

### 9.2 Memory Management
- Monitor heap usage with `ESP.getFreeHeap()`
- Use static buffers where possible
- Avoid dynamic allocation in audio callbacks
- Consider PSRAM if upgrading to ESP32-S3 variant

### 9.3 Audio Quality
- Use PDM mode for SPM1423 microphone
- Enable DC offset correction in ESPHome config
- Test sample rates: start with 16 kHz, try 8 kHz if CPU-constrained
- Implement noise gate to ignore low-level background

### 9.4 Testing Strategy
1. Verify microphone works (voice_assistant component)
2. Capture raw audio and visualize spectrum
3. Test FFT implementation with known tones
4. Validate with actual appliance beeps
5. Test in various noise conditions

### 9.5 ESPHome Integration
- Use external_components with GitHub repository
- Expose configuration via YAML (frequency ranges, thresholds)
- Publish as binary_sensor for Home Assistant
- Add optional sensor for confidence/magnitude
- Implement proper component lifecycle (setup, loop, cleanup)

---

## 10. Critical References

### Hardware
- ESP32-PICO-D4 Datasheet: https://www.espressif.com/sites/default/files/documentation/esp32-pico-d4_datasheet_en.pdf
- M5Stack Atom Echo Docs: https://docs.m5stack.com/en/atom/atomecho
- SPM1423 Microphone Datasheet: (search for "SPM1423 Knowles datasheet")

### ESPHome
- I2S Audio Component: https://esphome.io/components/i2s_audio/
- I2S Microphone: https://esphome.io/components/microphone/i2s_audio/
- External Components: https://esphome.io/components/external_components.html
- M5Stack Atom Echo Config: https://github.com/esphome/wake-word-voice-assistants/blob/main/m5stack-atom-echo/m5stack-atom-echo.yaml

### Libraries & Frameworks
- SoundAnalyzer (FFT/MFCC): https://github.com/MichielFromNL/SoundAnalyzer
- ESP-DSP (Espressif DSP): https://github.com/espressif/esp-dsp
- ESP-SR (Speech Recognition): https://github.com/espressif/esp-sr
- ESPHome Audio (ADF wrapper): https://github.com/gnumpi/esphome_audio

### Machine Learning
- Edge Impulse Docs: https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/audio/audio-classification
- TFLite Micro on ESP32: https://www.teachmemicro.com/getting-started-with-tensorflow-lite-on-esp32-with-voice-activity-detection-project/
- ESP32-S3 TFLite Guide: https://dev.to/zediot/esp32-s3-tensorflow-lite-micro-a-practical-guide-to-local-wake-word-edge-ai-inference-5540

### Example Projects
- ESP32 Audio Classification: https://forum.edgeimpulse.com/t/esp-32-audio-classification-inmp441/3439
- Offline Voice Recognition: https://www.hackster.io/ElectroScopeArchive/offline-esp32-voice-recognition-with-edge-impulse-fc93c9
- Portable Sound Analyzer: https://www.instructables.com/Portable-Sound-Analyzer-on-ESP32/

---

## 11. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Memory overflow | Medium | High | Start with FFT, profile carefully, use static buffers |
| Insufficient CPU | Low | Medium | Use lower sample rate (8kHz), optimize algorithms |
| PDM config issues | Medium | High | Follow existing Atom Echo configs, test thoroughly |
| Poor detection accuracy | Medium | Medium | Collect diverse training data, tune thresholds |
| ESPHome integration complexity | Low | Medium | Use external_components, follow examples |
| Power consumption | Low | Low | Optimize CPU frequency, consider sleep modes |

---

## 12. Conclusion & Next Steps

### Feasibility: ✅ HIGHLY FEASIBLE

The M5Stack Atom Echo (ESP32-PICO-D4) is **well-suited for beep detection** with the following caveats:

**Strengths**:
- Sufficient memory (520 KB RAM, 4 MB flash)
- Dual-core CPU @ 240 MHz
- PDM microphone with ESPHome support
- Active community and examples
- Multiple implementation approaches

**Constraints**:
- PDM microphone requires special configuration
- Memory must be managed carefully
- Cannot run Bluetooth + audio + ML simultaneously
- ~150-250ms latency for ML-based detection

### Recommended Approach

**PRIMARY: FFT-Based Detection (Non-ML)**
- Start here for simplest, fastest solution
- Perfect for detecting beeps at known frequencies
- 50-100ms latency
- ~150 KB RAM total

**FALLBACK: Edge Impulse ML Model**
- Use if FFT approach insufficient
- Handles complex beep patterns
- Robust to noise
- 150-250ms latency
- ~250-300 KB RAM

### Immediate Next Steps

1. **Hardware Setup**: Configure M5Stack Atom Echo with ESPHome I2S microphone
2. **Baseline Test**: Verify audio capture using voice_assistant component
3. **FFT Prototype**: Implement simple frequency detection with SoundAnalyzer
4. **Validation**: Test with target appliance beeps
5. **Decision Point**: Evaluate if FFT sufficient or ML needed

### Success Criteria

- Detect appliance beeps with >90% accuracy
- <500ms total latency (acceptable for appliance monitoring)
- Stable operation for 24+ hours
- <150 KB free RAM during operation
- Home Assistant integration via binary_sensor

---

**Document Prepared By**: Hardware & Audio Research Specialist
**For**: ESPHome Beeping Sensor Project
**Status**: Ready for Implementation Planning
