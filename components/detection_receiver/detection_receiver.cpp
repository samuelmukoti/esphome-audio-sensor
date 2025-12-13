#include "detection_receiver.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"

#include <cstring>
#include <errno.h>

namespace esphome {
namespace detection_receiver {

static const char *TAG = "detection_receiver";

bool DetectionReceiverComponent::init_socket() {
  // Create UDP socket
  this->sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (this->sock_ < 0) {
    ESP_LOGE(TAG, "Failed to create socket: %d", errno);
    return false;
  }

  // Set socket to non-blocking
  int flags = fcntl(this->sock_, F_GETFL, 0);
  if (flags < 0 || fcntl(this->sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
    ESP_LOGE(TAG, "Failed to set non-blocking: %d", errno);
    close(this->sock_);
    this->sock_ = -1;
    return false;
  }

  // Allow address reuse
  int opt = 1;
  setsockopt(this->sock_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Bind to port
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(this->port_);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);

  if (bind(this->sock_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    ESP_LOGE(TAG, "Failed to bind to port %d: %d", this->port_, errno);
    close(this->sock_);
    this->sock_ = -1;
    return false;
  }

  ESP_LOGI(TAG, "UDP socket bound to port %d", this->port_);
  return true;
}

void DetectionReceiverComponent::close_socket() {
  if (this->sock_ >= 0) {
    close(this->sock_);
    this->sock_ = -1;
  }
  this->socket_ready_ = false;
}

void DetectionReceiverComponent::setup() {
  ESP_LOGI(TAG, "Setting up Detection Receiver...");
  ESP_LOGI(TAG, "  Listening on UDP port: %d", this->port_);

  // Initialize socket
  if (this->init_socket()) {
    this->socket_ready_ = true;
    ESP_LOGI(TAG, "  UDP listener started successfully");
  } else {
    ESP_LOGE(TAG, "  Failed to start UDP listener!");
    this->mark_failed();
    return;
  }

  // Initialize sensors
  if (this->binary_sensor_ != nullptr) {
    this->binary_sensor_->publish_state(false);
  }
  if (this->confidence_sensor_ != nullptr) {
    this->confidence_sensor_->publish_state(0.0f);
  }
  if (this->detection_count_sensor_ != nullptr) {
    this->detection_count_sensor_->publish_state(0);
  }

  ESP_LOGI(TAG, "Detection Receiver ready");
}

void DetectionReceiverComponent::loop() {
  if (!this->socket_ready_ || this->sock_ < 0) {
    return;
  }

  // Check for incoming packets (non-blocking)
  uint8_t buffer[16];
  struct sockaddr_in src_addr;
  socklen_t src_addr_len = sizeof(src_addr);

  ssize_t len = recvfrom(this->sock_, buffer, sizeof(buffer), 0,
                         (struct sockaddr *)&src_addr, &src_addr_len);

  if (len > 0) {
    // Received a packet
    if (len >= 5) {
      this->process_packet(buffer, len);
      this->last_packet_time_ = millis();
    }
  } else if (len < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
    // Real error (not just "no data available")
    ESP_LOGW(TAG, "recvfrom error: %d", errno);
  }

  // Auto-clear detection if no packets received for a while
  // This handles the case where the server stops sending packets
  if (this->last_detection_state_ &&
      (millis() - this->last_packet_time_) > DETECTION_TIMEOUT_MS) {
    ESP_LOGD(TAG, "Detection timeout - clearing state");
    this->last_detection_state_ = false;
    if (this->binary_sensor_ != nullptr) {
      this->binary_sensor_->publish_state(false);
    }
    if (this->confidence_sensor_ != nullptr) {
      this->confidence_sensor_->publish_state(0.0f);
    }
  }
}

void DetectionReceiverComponent::process_packet(uint8_t *data, size_t len) {
  // Packet format: [1 byte: detected] + [4 bytes: confidence (float LE)]
  if (len < 5) {
    ESP_LOGW(TAG, "Packet too short: %d bytes", len);
    return;
  }

  bool detected = (data[0] != 0);

  // Extract confidence as little-endian float
  float confidence;
  memcpy(&confidence, &data[1], sizeof(float));

  ESP_LOGD(TAG, "Received detection: detected=%d, confidence=%.3f", detected, confidence);

  // Update confidence sensor (always update)
  if (this->confidence_sensor_ != nullptr) {
    this->confidence_sensor_->publish_state(confidence);
  }

  // Update binary sensor only on state change
  if (detected != this->last_detection_state_) {
    this->last_detection_state_ = detected;

    if (this->binary_sensor_ != nullptr) {
      this->binary_sensor_->publish_state(detected);
    }

    if (detected) {
      this->detection_count_++;
      ESP_LOGI(TAG, "BEEP DETECTED! confidence=%.3f, total=%d", confidence, this->detection_count_);

      if (this->detection_count_sensor_ != nullptr) {
        this->detection_count_sensor_->publish_state(this->detection_count_);
      }
    }
  }
}

void DetectionReceiverComponent::reset_detection_count() {
  this->detection_count_ = 0;
  if (this->detection_count_sensor_ != nullptr) {
    this->detection_count_sensor_->publish_state(0);
  }
  ESP_LOGI(TAG, "Detection count reset");
}

}  // namespace detection_receiver
}  // namespace esphome
