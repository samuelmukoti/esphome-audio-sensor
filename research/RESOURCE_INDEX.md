# Resource Index: ESPHome Beep Detection Research

**Compiled**: 2025-12-10
**Purpose**: Centralized index of all external resources referenced in research

---

## Official Documentation

### Hardware Documentation

**ESP32-PICO-D4 (SoC)**
- Datasheet: https://www.espressif.com/sites/default/files/documentation/esp32-pico-d4_datasheet_en.pdf
- ESP32 Series: https://www.espressif.com/sites/default/files/documentation/esp32-pico_series_datasheet_en.pdf
- Technical Reference: https://www.espressif.com/en/support/documents/technical-documents

**M5Stack Atom Echo**
- Product Page: https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit
- Documentation: https://docs.m5stack.com/en/atom/atomecho
- GitHub: https://github.com/m5stack/ATOM-ECHO
- Schematic: Available on product page

**SPM1423 Microphone**
- Knowles SPM1423 PDM microphone (search manufacturer site for datasheet)
- Community notes: https://community.m5stack.com/topic/4928/spm1423-i2s-microphone-u089-quirks

### ESPHome Documentation

**Core Components**
- ESPHome Main: https://esphome.io
- I2S Audio: https://esphome.io/components/i2s_audio/
- I2S Microphone: https://esphome.io/components/microphone/i2s_audio/
- External Components: https://esphome.io/components/external_components.html
- Custom Components: https://esphome.io/custom/custom_component.html

**API Reference**
- I2SAudioMicrophone Class: https://api-docs.esphome.io/classesphome_1_1i2s__audio_1_1_i2_s_audio_microphone

**Example Configurations**
- M5Stack Atom Echo (Official): https://github.com/esphome/wake-word-voice-assistants/blob/main/m5stack-atom-echo/m5stack-atom-echo.yaml
- Wake Word Voice Assistants: https://github.com/esphome/wake-word-voice-assistants

### ESP-IDF Documentation

**Audio Peripherals**
- I2S Driver: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html
- PDM Mode: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/peripherals/i2s.html#pdm-tx-and-rx-modes

---

## Libraries & Frameworks

### Audio Processing Libraries

**SoundAnalyzer** (RECOMMENDED for FFT)
- GitHub: https://github.com/MichielFromNL/SoundAnalyzer
- Features: FFT, MFCC, RMS, dBSPL, spectrum analysis
- Platform: Arduino/ESP32
- Performance: ~20ms for 1024 samples
- License: Open source
- Documentation: In repository README

**ESP-DSP** (Espressif Official)
- GitHub: https://github.com/espressif/esp-dsp
- Features: Optimized FFT, filters, matrix operations
- Platform: ESP-IDF native
- Performance: Assembly-optimized
- Documentation: https://docs.espressif.com/projects/esp-dsp/en/latest/

**ESP-SR** (Speech Recognition)
- GitHub: https://github.com/espressif/esp-sr
- Features: AFE, VAD, FFT, MFCC, wake word detection
- Platform: ESP-IDF
- Use Case: Speech/sound recognition
- Documentation: In repository + Espressif news

**ESPHome Audio (ADF Wrapper)**
- GitHub: https://github.com/gnumpi/esphome_audio
- Features: Espressif ADF integration with ESPHome
- Platform: ESPHome external component
- Use Case: Advanced audio pipelines

### Machine Learning Frameworks

**Edge Impulse** (RECOMMENDED for ML)
- Website: https://edgeimpulse.com
- Documentation: https://docs.edgeimpulse.com
- Tutorial (Audio): https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/audio/audio-classification
- ESP32 Support: https://docs.edgeimpulse.com/docs/edge-ai-hardware/mcu/espressif-esp32
- Forum: https://forum.edgeimpulse.com
- Pricing: Free tier available (sufficient for this project)

**TensorFlow Lite Micro**
- GitHub: https://github.com/tensorflow/tflite-micro
- ESP32 Examples: https://github.com/espressif/tflite-micro-esp-examples
- Arduino Library: https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32
- Documentation: https://www.tensorflow.org/lite/microcontrollers
- Tutorial: https://www.teachmemicro.com/getting-started-with-tensorflow-lite-on-esp32-with-voice-activity-detection-project/

**Tasmota TFLite Implementation**
- Documentation: https://tasmota.github.io/docs/TFL/
- Features: Audio classification with MFE/MFCC
- Reference implementation for ESP32

---

## Community Resources

### Forums & Communities

**ESPHome**
- Discord: https://discord.gg/KhAMKrd
- GitHub Discussions: https://github.com/esphome/esphome/discussions
- GitHub Issues: https://github.com/esphome/issues
- Home Assistant Forum: https://community.home-assistant.io/c/esphome/32

**M5Stack**
- Community Forum: https://community.m5stack.com
- Discord: Check M5Stack website for invite
- GitHub: https://github.com/m5stack

**Edge Impulse**
- Forum: https://forum.edgeimpulse.com
- Discord: https://discord.gg/edgeimpulse
- Documentation: https://docs.edgeimpulse.com

**Espressif (ESP32)**
- Forum: https://www.esp32.com
- GitHub: https://github.com/espressif
- Discord: Check Espressif website

### Social & Blogs

**Reddit**
- r/homeassistant: https://reddit.com/r/homeassistant
- r/esp32: https://reddit.com/r/esp32
- r/esphome: https://reddit.com/r/esphome

---

## Example Projects & Tutorials

### Audio Classification on ESP32

**ESP32 + INMP441 Audio Classification**
- Edge Impulse Forum: https://forum.edgeimpulse.com/t/esp-32-audio-classification-inmp441/3439
- Platform: Edge Impulse + ESP32
- Use Case: General audio classification

**Faucet Detection with ESP32**
- Edge Impulse Forum: https://forum.edgeimpulse.com/t/audio-classification-using-esp32-and-inmp441-microphone-to-detect-faucet-on/17182
- Platform: Edge Impulse + ESP32 + INMP441
- Use Case: Water running detection

**Offline ESP32 Voice Recognition**
- Hackster.io: https://www.hackster.io/ElectroScopeArchive/offline-esp32-voice-recognition-with-edge-impulse-fc93c9
- Platform: Edge Impulse + ESP32
- Use Case: Voice commands

**Portable Sound Analyzer on ESP32**
- Instructables: https://www.instructables.com/Portable-Sound-Analyzer-on-ESP32/
- Platform: ESP32 + FFT
- Use Case: Frequency analysis

**Voice Command Recognition with ESP32 and TinyML**
- Tutorial: https://www.teachmemicro.com/voice-command-recognition-with-esp32-and-tinyml-using-edge-impulse/
- Platform: Edge Impulse + ESP32
- Use Case: Voice commands

**ESP32 TensorFlow Micro Speech**
- Tutorial: https://www.survivingwithandroid.com/esp32-tensorflow-micro-speech-i2s-external-microphone/
- Platform: TFLite Micro + ESP32 + I2S mic
- Use Case: Speech recognition

**Continuous Audio Sampling**
- Edge Impulse Docs: https://docs.edgeimpulse.com/docs/tutorials/advanced-inferencing/continuous-audio-sampling
- Use Case: Real-time audio monitoring

### M5Stack Atom Echo Specific

**M5Stack Atom Echo with ESPHome**
- Blog Post: https://niksa.dev/posts/m5stack-echo/
- Platform: ESPHome
- Use Case: Voice assistant setup

**M5Stack Atom Echo for Home Assistant**
- Tutorial: https://peyanski.com/m5-atom-echo-for-home-assistant-wake-word-control/
- Platform: ESPHome + Home Assistant
- Use Case: Wake word control

**$13 Voice Assistant for Home Assistant**
- Official HA Guide: https://www.home-assistant.io/voice_control/thirteen-usd-voice-remote/
- Platform: ESPHome + M5Stack Atom Echo
- Use Case: Budget voice assistant

### ESP32 Audio Development

**ESP32 FFT on Audio**
- Blog: http://www.robinscheibler.org/2017/12/12/esp32-fft.html
- Platform: ESP32 + I2S
- Use Case: Real-time FFT

**ESP32 FFT Audio LEDs**
- GitHub: https://github.com/debsahu/ESP32_FFT_Audio_LEDs
- Platform: ESP32 + MSGEQ7
- Use Case: Audio visualization

**ESP32 Audio Sampling for ML**
- GitHub: https://github.com/happychriss/edgeML_esp32_audio_sampling
- Platform: ESP32 + Edge Impulse
- Use Case: Number recognition

---

## Technical Articles & Guides

### TensorFlow Lite on ESP32

**Getting Started with TFLite on ESP32**
- Guide: https://www.teachmemicro.com/getting-started-with-tensorflow-lite-on-esp32-with-voice-activity-detection-project/
- Topics: Setup, VAD project, deployment

**TinyML with ESP32**
- Tutorial: https://www.teachmemicro.com/tinyml-with-esp32-tutorial/
- Topics: Model training, optimization, deployment

**ESP32-S3 + TFLite Micro Guide**
- Blog: https://dev.to/zediot/esp32-s3-tensorflow-lite-micro-a-practical-guide-to-local-wake-word-edge-ai-inference-5540
- Detailed: https://zediot.com/blog/esp32-s3-tensorflow-lite-micro/
- Topics: Wake word detection, edge AI

**TFLite on ESP32 Overview**
- Guide: https://openelab.io/blogs/learn/tensorflow-lite-on-esp32
- Topics: Setup, limitations, examples

**Integer vs Float on ESP32-S3**
- Medium: https://medium.com/@sfarrukhm/integer-vs-float-performance-on-the-esp32-s3-why-tinyml-loves-quantization-227eca11bd35
- Topics: Quantization benefits, performance

### Audio Processing Concepts

**Sound Recognition Basics**
- ESP32 Forum: https://esp32.com/viewtopic.php?t=1841
- Topics: Spectrum analysis, FFT, pattern matching

**FFT Tone Detection**
- Adafruit: https://learn.adafruit.com/fft-fun-with-fourier-transforms/tone-input
- Topics: Frequency analysis, tone detection

**Speech Recognition and Speech-to-Intent**
- Seeed Wiki: https://wiki.seeedstudio.com/Wio-Terminal-TinyML-TFLM-3/
- Topics: Audio preprocessing, ML inference

### ESP-IDF Development

**ESP32 Audio Development Overview**
- Espressif News: https://www.espressif.com/en/news/ESP-Skainet_Released
- Topics: ESP-Skainet framework, wake word detection

**ESP Audio Development Framework**
- Forum Discussion: https://github.com/esphome/feature-requests/issues/1882
- Topics: Audio dev boards, ADF integration

---

## Tools & Utilities

### Development Tools

**ESPHome**
- Installation: `pip install esphome`
- CLI Tool: `esphome`
- Web Dashboard: Built-in
- Documentation: https://esphome.io

**PlatformIO**
- Website: https://platformio.org
- VS Code Extension: Available
- Use Case: Advanced ESP32 development

**Arduino IDE**
- Website: https://www.arduino.cc/en/software
- ESP32 Board Support: Via Board Manager
- Use Case: Simple prototyping

### Audio Analysis Tools

**Audacity**
- Website: https://www.audacityteam.org
- Use Case: Record and analyze audio samples
- Features: Spectrum analysis, frequency visualization

**Sonic Visualiser**
- Website: https://www.sonicvisualiser.org
- Use Case: Advanced audio analysis
- Features: Spectrogram, pitch tracking

### Home Assistant

**Installation**
- Website: https://www.home-assistant.io
- ESPHome Integration: Built-in
- Documentation: https://www.home-assistant.io/integrations/esphome/

---

## Hardware Suppliers

### M5Stack Atom Echo

**Official Store**
- M5Stack Store: https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit
- Price: ~$13 USD

**Distributors**
- The Pi Hut: https://thepihut.com/products/atom-echo-smart-speaker-dev-kit
- DigiKey: https://www.digikey.com (search: M5Stack Atom Echo)
- Mouser: https://www.mouser.com (search: M5Stack Atom Echo)
- RobotShop: https://www.robotshop.com/products/m5stack-atom-echo-smart-speaker-dev-kit

### Alternative Microphones (for custom builds)

**INMP441** (I2S MEMS Microphone)
- Type: I2S digital microphone
- Use Case: External mic for ESP32
- Suppliers: Adafruit, SparkFun, AliExpress

**ICS-43434** (I2S Microphone)
- Type: I2S digital microphone
- Features: Low noise, high SNR
- Suppliers: TDK/InvenSense distributors

**MAX9814** (Analog with AGC)
- Type: Analog microphone with auto gain
- Features: Automatic gain control
- Suppliers: Adafruit, SparkFun

---

## Code Repositories

### ESPHome Repositories

**Official ESPHome**
- Main: https://github.com/esphome/esphome
- Issues: https://github.com/esphome/issues
- Feature Requests: https://github.com/esphome/feature-requests
- Voice Assistants: https://github.com/esphome/wake-word-voice-assistants
- Audio Libraries: https://github.com/esphome/esp-audio-libs

**Community Components**
- ESPHome Audio (ADF): https://github.com/gnumpi/esphome_audio
- Custom Examples: https://github.com/thegroove/esphome-custom-component-examples
- External Components: https://github.com/jesserockz/esphome-external-component-examples

### Audio Processing Libraries

**SoundAnalyzer**
- GitHub: https://github.com/MichielFromNL/SoundAnalyzer
- License: Open source
- Features: FFT, MFCC, RMS, spectrum

**ESP-DSP**
- GitHub: https://github.com/espressif/esp-dsp
- License: Apache 2.0
- Features: Optimized DSP functions

**ESP-SR**
- GitHub: https://github.com/espressif/esp-sr
- License: Espressif proprietary
- Features: Speech recognition

### Machine Learning

**TFLite Micro for ESP32**
- Examples: https://github.com/espressif/tflite-micro-esp-examples
- Arduino: https://github.com/tanakamasayuki/Arduino_TensorFlowLite_ESP32

**Speech Recognition Projects**
- ESP32 Speech: https://github.com/shaurya0406/ESP32_Speech_Recognition_TensorFlow
- Audio Sampling: https://github.com/happychriss/edgeML_esp32_audio_sampling

---

## Research Papers & Academic Resources

### Audio Classification

**Mel-Frequency Cepstral Coefficients (MFCC)**
- Wikipedia: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum
- Use Case: Understanding audio features

**Fast Fourier Transform (FFT)**
- Wikipedia: https://en.wikipedia.org/wiki/Fast_Fourier_transform
- Use Case: Frequency domain analysis

**Sound Event Detection**
- Papers: Search Google Scholar for "sound event detection embedded systems"
- Use Case: Academic background

### TinyML & Edge AI

**TinyML Book**
- O'Reilly: "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers"
- Authors: Pete Warden, Daniel Situnayake

**Edge Impulse Blog**
- Blog: https://www.edgeimpulse.com/blog
- Topics: TinyML tutorials, case studies

---

## Videos & Courses

### YouTube Tutorials

**ESPHome Voice Assistant**
- Search: "ESPHome M5Stack Atom Echo voice assistant"
- Multiple creators with step-by-step guides

**ESP32 Audio Processing**
- Search: "ESP32 audio classification TensorFlow"
- Various tutorials on ML audio projects

**Edge Impulse ESP32**
- Edge Impulse Channel: Official tutorials
- Topics: Audio classification, deployment

### Online Courses

**TinyML Specialization** (Coursera)
- Provider: edX / Harvard
- Topics: ML on microcontrollers

**Edge Impulse University**
- Website: https://docs.edgeimpulse.com/docs/
- Free: Yes
- Topics: End-to-end ML deployment

---

## Standards & Specifications

### Audio Standards

**I2S (Inter-IC Sound)**
- Spec: Philips I2S bus specification
- Use Case: Digital audio interface

**PDM (Pulse Density Modulation)**
- Spec: Digital microphone interface
- Use Case: MEMS microphones (like SPM1423)

**WAV / PCM Audio**
- Format: Standard audio file formats
- Use Case: Audio data representation

### Communication Protocols

**MQTT** (for Home Assistant)
- Spec: MQTT v3.1.1 / v5
- Use Case: IoT messaging

**ESPHome API**
- Protocol: Native API
- Use Case: Home Assistant communication

---

## Datasheets Index

### ICs & Components

1. **ESP32-PICO-D4**: Espressif website (listed above)
2. **SPM1423**: Knowles manufacturer site
3. **SK6812 LED**: LED datasheet (M5Stack schematic)

### Development Boards

1. **M5Stack Atom Echo**: M5Stack documentation (listed above)
2. **M5Stack Atom Base**: M5Stack website

---

## Quick Link Summary (Top 10 Essential)

For quick access, here are the 10 most critical links:

1. **ESPHome I2S Audio**: https://esphome.io/components/i2s_audio/
2. **M5Stack Atom Echo Config**: https://github.com/esphome/wake-word-voice-assistants/blob/main/m5stack-atom-echo/m5stack-atom-echo.yaml
3. **SoundAnalyzer (FFT)**: https://github.com/MichielFromNL/SoundAnalyzer
4. **Edge Impulse (ML)**: https://edgeimpulse.com
5. **ESP32 Datasheet**: https://www.espressif.com/sites/default/files/documentation/esp32-pico-d4_datasheet_en.pdf
6. **ESPHome Discord**: https://discord.gg/KhAMKrd
7. **M5Stack Docs**: https://docs.m5stack.com/en/atom/atomecho
8. **ESP-DSP Library**: https://github.com/espressif/esp-dsp
9. **TFLite Micro ESP32**: https://github.com/espressif/tflite-micro-esp-examples
10. **Audio Tutorial**: https://docs.edgeimpulse.com/docs/tutorials/end-to-end-tutorials/audio/audio-classification

---

## Resource Validation

All links verified as of: **2025-12-10**

**Note**: External links may change over time. If a link is broken:
1. Search for the resource name + "github" or "documentation"
2. Check archived versions via Wayback Machine
3. Contact research team for updated links

---

**Compiled By**: Hardware & Audio Research Specialist
**Total Resources**: 100+ links across 10 categories
**Last Updated**: 2025-12-10
**Status**: Complete & Verified
