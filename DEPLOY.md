# Deployment Guide

This guide covers deploying the beep detection system with:
- **ESP32 firmware** (via ESPHome)
- **Beep Detection Server** (via Docker)

## Architecture

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
       │                                    │
       v                                    │ :8080
┌──────────────┐                           │
│ Home         │ <─────────────────────────┘
│ Assistant    │      (optional)
└──────────────┘
```

## Quick Start

### 1. Deploy the Detection Server

#### Option A: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/samuelmukoti/esphome-audio-sensor.git
cd esphome-audio-sensor

# Start the server
docker-compose up -d

# View logs
docker-compose logs -f beep-detector
```

#### Option B: Pull from GitHub Container Registry

```bash
# Pull the latest image
docker pull ghcr.io/samuelmukoti/esphome-audio-sensor/beep-detector:latest

# Run the container
docker run -d --name beep-detector \
  -p 8080:8080 \
  -p 5050:5050/udp \
  -p 5001:5001/udp \
  -v beep-data:/app/recordings \
  ghcr.io/samuelmukoti/esphome-audio-sensor/beep-detector:latest
```

#### Option C: Run directly with Python

```bash
cd server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python audio_server.py --port 5050 --web-port 8080 --confidence-threshold 0.7
```

### 2. Flash ESP32 Firmware

#### Using ESPHome Dashboard (Recommended)

1. Copy `esphome-atom-d4d5d0.yaml` to your ESPHome config directory
2. Create `secrets.yaml` with your credentials:
   ```yaml
   wifi_ssid: "your_wifi_ssid"
   wifi_password: "your_wifi_password"
   api_encryption_key: "your_32_byte_base64_key"
   ota_password: "your_ota_password"
   ```
3. Update the server IP in the YAML:
   ```yaml
   audio_streamer:
     server_ip: "YOUR_SERVER_IP"  # Change this!
     server_port: 5050
   ```
4. Install via ESPHome dashboard

#### Using CLI

```bash
# Install ESPHome
pip install esphome

# Create secrets.yaml (see above)

# Build and flash
esphome run esphome-atom-d4d5d0.yaml
```

### 3. Access the Dashboard

Open `http://YOUR_SERVER_IP:8080` to access:
- Live audio visualization
- Detection confidence meter
- Training mode controls
- Sample labeling interface
- Model retraining

## Ports

| Port | Protocol | Description |
|------|----------|-------------|
| 8080 | TCP | Web dashboard |
| 5050 | UDP | Audio stream from ESP32 |
| 5001 | UDP | Detection results to ESP32 |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| AUDIO_PORT | 5050 | UDP port for audio stream |
| WEB_PORT | 8080 | HTTP port for dashboard |
| CONFIDENCE_THRESHOLD | 0.7 | Detection threshold (0.0-1.0) |

## Custom Components

The ESP32 firmware uses these custom ESPHome components:

- **audio_streamer**: Streams PDM microphone audio via UDP
- **detection_receiver**: Receives detection results from server
- **beep_detector_nn**: (Optional) On-device neural network inference

### Using Components in Your Own Project

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

## Training Your Own Model

1. Enable training mode in the dashboard
2. Collect samples:
   - Click "Mark Beep NOW" when you hear a beep
   - Label detected sounds in "Pending Review"
3. Click "Retrain Model"
4. Test and iterate

## Troubleshooting

### No audio stream received
- Check ESP32 is connected to WiFi
- Verify server IP in ESP32 config
- Check UDP port 5050 is open
- Check firewall rules

### High false positive rate
- Collect more negative (background) samples
- Increase confidence threshold
- Retrain model with balanced dataset

### Docker container won't start
```bash
# Check logs
docker logs beep-detector

# Verify ports are available
netstat -an | grep -E "5050|5001|8080"
```

## GitHub Actions

The repository includes workflows for:
- **Docker build**: Automatically builds and pushes to GHCR on push to main
- **ESPHome build**: Validates and builds firmware on changes

## License

MIT License
