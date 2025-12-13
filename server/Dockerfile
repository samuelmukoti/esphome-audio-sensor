# Beep Detection Server
# Runs neural network inference on audio streamed from ESP32

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY audio_server.py .
COPY models/ models/

# Create directories for data persistence
RUN mkdir -p /app/recordings/labeled_data/audio /app/models

# Expose ports
# 5050: UDP audio stream from ESP32
# 5001: UDP detection results to ESP32
# 8080: Web dashboard
EXPOSE 5050/udp
EXPOSE 5001/udp
EXPOSE 8080

# Environment variables with defaults
ENV AUDIO_PORT=5050
ENV WEB_PORT=8080
ENV CONFIDENCE_THRESHOLD=0.7

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${WEB_PORT}/api/status || exit 1

# Run the server
CMD ["python", "-u", "audio_server.py", \
     "--port", "5050", \
     "--web-port", "8080", \
     "--confidence-threshold", "0.7"]
