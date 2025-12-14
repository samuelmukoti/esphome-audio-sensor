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

## Development Workflow

This section documents the actual workflow for developing and deploying this system.

### Environment Overview

| Component | Location | Description |
|-----------|----------|-------------|
| Development Mac | Local | ESPHome builds, model training, code editing |
| Docker Server | `192.168.86.10` | Runs inference server (Intel Celeron N3050) |
| ESP32 Device | `192.168.86.71` | M5Stack Atom Echo with PDM mic |

### Prerequisites

```bash
# On development machine
pip install esphome tensorflow

# Server should have Docker and docker-compose installed
```

### 1. Deploy Docker Container via SSH

The server runs on a low-power Intel Celeron N3050 which **lacks AVX instructions**, so we use TensorFlow Lite instead of full TensorFlow.

```bash
# SSH into the server
ssh sam@192.168.86.10

# Create deployment directory
mkdir -p /home/sam/docker/esphome-audio-sensor

# Exit back to dev machine
exit
```

#### Sync files to server:
```bash
# From project root on dev machine
rsync -avz --exclude '.git' --exclude '.esphome' --exclude 'venv' --exclude '.venv' --exclude '__pycache__' \
  server/ sam@192.168.86.10:/home/sam/docker/esphome-audio-sensor/server/

scp docker-compose.yml sam@192.168.86.10:/home/sam/docker/esphome-audio-sensor/
```

#### Build and start container:
```bash
ssh sam@192.168.86.10 "cd /home/sam/docker/esphome-audio-sensor && docker-compose up -d --build"
```

#### Check logs:
```bash
ssh sam@192.168.86.10 "docker logs --tail 50 beep-detector"
```

### 2. Convert Keras Model to TFLite

For servers without AVX instructions (older Intel Atom, Celeron, etc.), convert the Keras model to TFLite:

```bash
# On dev machine with TensorFlow installed
cd /path/to/esphome-audio-sensor

source .venv/bin/activate  # or create venv with: python -m venv .venv

python3 << 'EOF'
import tensorflow as tf

# Load the ACTIVE trained model (not the default one!)
model = tf.keras.models.load_model('tools/models/beep_detector_active.keras')
print(f"Loaded model: {model.input_shape} -> {model.output_shape}")

# Convert to TFLite with optimization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save to server/models
with open('server/models/beep_detector.tflite', 'wb') as f:
    f.write(tflite_model)
print(f"Saved TFLite model ({len(tflite_model)} bytes)")
EOF
```

**Important:** Always convert `beep_detector_active.keras` (the trained model), not `beep_detector.keras` (the default/untrained model).

### 3. Deploy Updated Model

After retraining or converting a new model:

```bash
# Copy the new TFLite model to server
scp server/models/beep_detector.tflite sam@192.168.86.10:/home/sam/docker/esphome-audio-sensor/server/models/

# Restart container to load new model
ssh sam@192.168.86.10 "docker restart beep-detector"

# Verify model loaded
ssh sam@192.168.86.10 "docker logs --tail 20 beep-detector 2>&1 | head -10"
```

Expected output:
```
[INFO] Using TFLite runtime for inference (no AVX required)
  TFLite model loaded: models/beep_detector.tflite
```

### 4. Build and Flash ESP32 Firmware

#### Update server IP in config:
Edit `esphome-atom-d4d5d0.yaml`:
```yaml
audio_streamer:
  target_ip: "192.168.86.10"  # Your server IP
  target_port: 5050

detection_receiver:
  listen_port: 5001
```

#### Flash via USB (first time):
```bash
# Connect ESP32 via USB
esphome run esphome-atom-d4d5d0.yaml --device /dev/cu.usbserial-*
```

#### Flash via OTA (subsequent updates):
```bash
# ESP32 already on network - uses OTA automatically
esphome run esphome-atom-d4d5d0.yaml
```

ESPHome will detect the device on the network and offer OTA upload.

### 5. Monitor the System

#### Server logs (detection activity):
```bash
ssh sam@192.168.86.10 "docker logs -f beep-detector 2>&1"
```

#### ESP32 logs (streaming/receiving):
```bash
esphome logs esphome-atom-d4d5d0.yaml
```

#### Web dashboard:
Open `http://192.168.86.10:8080` in browser.

### 6. Training Workflow

1. **Enable training mode** in web dashboard
2. **Collect samples** - mark beeps and label pending detections
3. **Retrain model** via dashboard (creates new `beep_detector_active.keras`)
4. **Convert to TFLite** (see step 2 above)
5. **Deploy to server** (see step 3 above)

### Quick Reference Commands

```bash
# Check container status
ssh sam@192.168.86.10 "docker-compose -f /home/sam/docker/esphome-audio-sensor/docker-compose.yml ps"

# View real-time detection output
ssh sam@192.168.86.10 "docker logs -f beep-detector 2>&1 | grep -E 'conf=|BEEP'"

# Restart container
ssh sam@192.168.86.10 "docker restart beep-detector"

# Rebuild container (after code changes)
ssh sam@192.168.86.10 "cd /home/sam/docker/esphome-audio-sensor && docker-compose up -d --build"

# Flash ESP32 firmware
esphome run esphome-atom-d4d5d0.yaml

# Monitor ESP32 logs
esphome logs esphome-atom-d4d5d0.yaml
```

### Troubleshooting

#### TensorFlow crashes with SIGILL (exit code 132)
- **Cause**: Server CPU lacks AVX instructions
- **Fix**: Use `tflite-runtime` instead of `tensorflow` in requirements.txt

#### NumPy version error with TFLite
- **Error**: "module compiled using NumPy 1.x cannot run in NumPy 2.x"
- **Fix**: Pin numpy in requirements.txt: `numpy>=1.24.0,<2.0.0`

#### False positives after deployment
- **Cause**: Wrong model deployed (default vs trained)
- **Fix**: Convert `beep_detector_active.keras`, not `beep_detector.keras`

#### No audio packets received
- Check ESP32 is streaming: `esphome logs esphome-atom-d4d5d0.yaml`
- Verify server IP in ESP32 config matches actual server
- Check UDP port 5050 is open on server firewall

## License

MIT License
