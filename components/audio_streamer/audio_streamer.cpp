#include "audio_streamer.h"
#include "esphome/core/log.h"
#include "esphome/core/hal.h"

#include <cstring>
#include <arpa/inet.h>

namespace esphome {
namespace audio_streamer {

static const char *TAG = "audio_streamer";

void AudioStreamerComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Audio Streamer...");
  ESP_LOGCONFIG(TAG, "  Target: %s:%d", this->target_ip_.c_str(), this->target_port_);
  ESP_LOGCONFIG(TAG, "  Sample Rate: %d Hz", this->sample_rate_);
  ESP_LOGCONFIG(TAG, "  Chunk Size: %d bytes", this->chunk_size_);
  ESP_LOGCONFIG(TAG, "  Auto-start: %s", this->enabled_ ? "yes" : "no");

  // Reserve buffer space (sequence number + audio data)
  this->send_buffer_.reserve(4 + this->chunk_size_);

  // Register audio callback with microphone
  if (this->microphone_ != nullptr) {
    this->microphone_->add_data_callback([this](const std::vector<uint8_t> &data) {
      if (this->streaming_active_) {
        this->stream_audio_data(data);
      }
    });
    ESP_LOGI(TAG, "Audio callback registered");
  } else {
    ESP_LOGE(TAG, "Microphone is NULL - cannot stream audio!");
    return;
  }

  // Auto-start if configured
  if (this->enabled_) {
    this->start_streaming();
  }

  this->last_stats_time_ = millis();
}

void AudioStreamerComponent::loop() {
  // Periodic stats logging
  uint32_t now = millis();
  if (this->streaming_active_ && (now - this->last_stats_time_ >= STATS_INTERVAL_MS)) {
    float kbps = (this->bytes_sent_ * 8.0f) / (STATS_INTERVAL_MS / 1000.0f) / 1000.0f;
    ESP_LOGI(TAG, "Streaming stats: %d packets, %.1f kbps to %s:%d",
             this->packets_sent_, kbps, this->target_ip_.c_str(), this->target_port_);
    this->reset_stats();
    this->last_stats_time_ = now;
  }
}

bool AudioStreamerComponent::init_socket() {
  // Create UDP socket
  this->sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (this->sock_ < 0) {
    ESP_LOGE(TAG, "Failed to create socket: errno %d", errno);
    return false;
  }

  // Set up destination address
  memset(&this->dest_addr_, 0, sizeof(this->dest_addr_));
  this->dest_addr_.sin_family = AF_INET;
  this->dest_addr_.sin_port = htons(this->target_port_);

  // Convert IP address string to binary
  if (inet_pton(AF_INET, this->target_ip_.c_str(), &this->dest_addr_.sin_addr) <= 0) {
    ESP_LOGE(TAG, "Invalid IP address: %s", this->target_ip_.c_str());
    close(this->sock_);
    this->sock_ = -1;
    return false;
  }

  ESP_LOGI(TAG, "UDP socket created for %s:%d", this->target_ip_.c_str(), this->target_port_);
  return true;
}

void AudioStreamerComponent::close_socket() {
  if (this->sock_ >= 0) {
    close(this->sock_);
    this->sock_ = -1;
  }
}

void AudioStreamerComponent::stream_audio_data(const std::vector<uint8_t> &data) {
  if (data.empty() || this->sock_ < 0) {
    return;
  }

  // Build UDP packet: [4-byte sequence number][audio data]
  this->send_buffer_.clear();

  // Add sequence number (little-endian)
  uint32_t seq = this->sequence_number_++;
  this->send_buffer_.push_back(seq & 0xFF);
  this->send_buffer_.push_back((seq >> 8) & 0xFF);
  this->send_buffer_.push_back((seq >> 16) & 0xFF);
  this->send_buffer_.push_back((seq >> 24) & 0xFF);

  // Add audio data
  this->send_buffer_.insert(this->send_buffer_.end(), data.begin(), data.end());

  // Send UDP packet
  ssize_t sent = sendto(this->sock_, this->send_buffer_.data(), this->send_buffer_.size(), 0,
                        (struct sockaddr *)&this->dest_addr_, sizeof(this->dest_addr_));

  if (sent > 0) {
    this->packets_sent_++;
    this->bytes_sent_ += sent;
  } else {
    ESP_LOGW(TAG, "Failed to send UDP packet: errno %d", errno);
  }
}

void AudioStreamerComponent::start_streaming() {
  if (this->streaming_active_) {
    ESP_LOGW(TAG, "Streaming already active");
    return;
  }

  if (this->microphone_ == nullptr) {
    ESP_LOGE(TAG, "Cannot start streaming - no microphone configured");
    return;
  }

  ESP_LOGI(TAG, "Starting audio stream to %s:%d",
           this->target_ip_.c_str(), this->target_port_);

  // Initialize the UDP socket
  if (!this->init_socket()) {
    ESP_LOGE(TAG, "Failed to initialize socket");
    return;
  }

  // Start the microphone if not already running
  this->microphone_->start();

  this->streaming_active_ = true;
  this->sequence_number_ = 0;
  this->reset_stats();
  this->last_stats_time_ = millis();

  ESP_LOGI(TAG, "Audio streaming started");
}

void AudioStreamerComponent::stop_streaming() {
  if (!this->streaming_active_) {
    ESP_LOGW(TAG, "Streaming not active");
    return;
  }

  ESP_LOGI(TAG, "Stopping audio stream");

  this->streaming_active_ = false;
  this->close_socket();

  // Note: We don't stop the microphone here in case beep_detector is also using it

  ESP_LOGI(TAG, "Audio streaming stopped (sent %d packets total)", this->packets_sent_);
}

void AudioStreamerComponent::reset_stats() {
  this->packets_sent_ = 0;
  this->bytes_sent_ = 0;
}

}  // namespace audio_streamer
}  // namespace esphome
