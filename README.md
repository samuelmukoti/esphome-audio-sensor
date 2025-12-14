# ESPHome Audio Beep Detector

Real-time beep detection system using ESP32 with neural network inference for Home Assistant integration.

## Overview

This project provides a beep detection system that:
- Streams audio from an ESP32 (M5Stack Atom Echo) via UDP
- Runs neural network inference on a server for accurate detection
- Sends detection results back to ESP32 for Home Assistant integration
- Includes a web dashboard for training and active learning

```
┌──────────────┐     UDP:5050      ┌─────────────────────┐
│    ESP32     │ ──────────────>   │   Beep Detection    │
│  (M5 Atom)   │                   │      Server         │
│              │ <──────────────   │   (Docker/Python)   │
│  - PDM Mic   │     UDP:5001      │                     │
│  - WiFi      │                   │  - Neural Network   │
└──────────────┘                   │  - Web Dashboard    │
       │                           │  - Active Learning  │
       │ API                       └─────────────────────┘
       v                                    │
┌──────────────┐                           │ :8080
│ Home         │ <─────────────────────────┘
│ Assistant    │
└──────────────┘
```

## Quick Start

### 1. Deploy the Detection Server

```bash
# Using Docker Compose (recommended)
git clone https://github.com/samuelmukoti/esphome-audio-sensor.git
cd esphome-audio-sensor
docker-compose up -d

# View logs
docker-compose logs -f beep-detector
```

Or pull from GitHub Container Registry:
```bash
docker pull ghcr.io/samuelmukoti/esphome-audio-sensor/beep-detector:latest
docker run -d --name beep-detector \
  -p 8080:8080 -p 5050:5050/udp -p 5001:5001/udp \
  ghcr.io/samuelmukoti/esphome-audio-sensor/beep-detector:latest
```

### 2. Flash ESP32 Firmware

1. Copy `esphome-atom-d4d5d0.yaml` to your ESPHome config directory
2. Create `secrets.yaml`:
   ```yaml
   wifi_ssid: "your_wifi_ssid"
   wifi_password: "your_wifi_password"
   api_encryption_key: "your_32_byte_base64_key"
   ota_password: "your_ota_password"
   ```
3. Update server IP in the YAML config
4. Flash via ESPHome dashboard or CLI:
   ```bash
   esphome run esphome-atom-d4d5d0.yaml
   ```

### 3. Access the Dashboard

Open `http://YOUR_SERVER_IP:8080` to access:
- Live audio visualization
- Detection confidence meter
- Training mode controls
- Sample labeling interface

## Components

### ESPHome Custom Components

| Component | Description |
|-----------|-------------|
| `audio_streamer` | Streams PDM microphone audio via UDP |
| `detection_receiver` | Receives detection results from server |
| `beep_detector` | Simple on-device energy detection |
| `beep_detector_nn` | On-device neural network (experimental) |

### Using Components in Your Project

```yaml
external_components:
  - source:
      type: git
      url: https://github.com/samuelmukoti/esphome-audio-sensor
      ref: main
    components: [audio_streamer, detection_receiver]

audio_streamer:
  id: audio_stream
  server_ip: "192.168.1.100"
  server_port: 5050
  sample_rate: 16000

detection_receiver:
  id: detector
  listen_port: 5001
  on_detection:
    - logger.log: "Beep detected!"
```

## Architecture

- **ESP32**: Captures audio via I2S PDM microphone, streams over UDP
- **Detection Server**: Python Flask app with TensorFlow model
- **Neural Network**: CNN trained on MFCC features (13 coefficients, 25 frames)
- **Active Learning**: Web interface for labeling samples and retraining

## Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 8080 | TCP | Web dashboard |
| 5050 | UDP | Audio stream from ESP32 |
| 5001 | UDP | Detection results to ESP32 |

## Training Your Own Model

1. Enable training mode in the dashboard
2. Collect samples:
   - Click "Mark Beep NOW" when you hear a beep
   - Label detected sounds in "Pending Review"
3. Click "Retrain Model"
4. Test and iterate

## Hardware

**M5Stack Atom Echo:**
- ESP32-PICO (240MHz dual-core)
- SPM1423 I2S MEMS microphone
- USB-C powered

**I2S Pin Configuration:**
- BCLK: GPIO 19
- LRCLK: GPIO 33
- DATA_IN: GPIO 22

## Project Structure

```
esphome-audio-sensor/
├── components/              # ESPHome custom components
│   ├── audio_streamer/      # UDP audio streaming
│   ├── detection_receiver/  # Receive detection results
│   ├── beep_detector/       # Simple energy detection
│   └── beep_detector_nn/    # On-device neural network
├── server/                  # Detection server
│   ├── audio_server.py      # Main server application
│   ├── Dockerfile           # Container definition
│   ├── models/              # Trained models
│   └── requirements.txt     # Python dependencies
├── tools/                   # Training and analysis tools
├── .github/workflows/       # CI/CD pipelines
├── docker-compose.yml       # Container orchestration
├── esphome-atom-d4d5d0.yaml # Example ESPHome config
└── DEPLOY.md                # Deployment guide
```

## CI/CD

GitHub Actions automatically:
- **Docker Build**: Builds multi-arch images (amd64/arm64) and pushes to GHCR
- **ESPHome Build**: Validates config and builds firmware artifacts

## License

MIT License
