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
  ESP_LOGCONFIG(TAG, "  Batch Count: %d reads", this->batch_count_);
  ESP_LOGCONFIG(TAG, "  Send Interval: %d ms (max %d pkt/s)", this->send_interval_ms_,
                1000 / (this->send_interval_ms_ > 0 ? this->send_interval_ms_ : 1));
  ESP_LOGCONFIG(TAG, "  Auto-start: %s", this->enabled_ ? "yes" : "no");

  // Reserve buffer space for batched data (sequence number + batched audio data)
  // Each mic read is ~512 bytes, so batch_count * 512 + 4 bytes for seq
  size_t max_batch_size = 4 + (this->batch_count_ * 1024);  // Extra headroom
  this->send_buffer_.reserve(max_batch_size);
  this->accumulation_buffer_.reserve(max_batch_size);

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
    float pps = (this->packets_sent_ * 1000.0f) / STATS_INTERVAL_MS;
    ESP_LOGI(TAG, "Streaming: %d sent (%.1f pkt/s), %d dropped, %.1f kbps to %s:%d",
             this->packets_sent_, pps, this->packets_dropped_, kbps,
             this->target_ip_.c_str(), this->target_port_);
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

  // Accumulate data for batching
  this->accumulation_buffer_.insert(this->accumulation_buffer_.end(), data.begin(), data.end());
  this->batch_counter_++;

  // Check if we should send now (batching complete or interval elapsed)
  uint32_t now = millis();
  bool batch_ready = (this->batch_counter_ >= this->batch_count_);
  bool interval_ready = (now - this->last_send_time_ >= this->send_interval_ms_);

  // Only send if batch is ready AND interval has passed
  if (!batch_ready || !interval_ready) {
    return;
  }

  // Build UDP packet: [4-byte sequence number][batched audio data]
  this->send_buffer_.clear();

  // Add sequence number (little-endian)
  uint32_t seq = this->sequence_number_++;
  this->send_buffer_.push_back(seq & 0xFF);
  this->send_buffer_.push_back((seq >> 8) & 0xFF);
  this->send_buffer_.push_back((seq >> 16) & 0xFF);
  this->send_buffer_.push_back((seq >> 24) & 0xFF);

  // Add accumulated audio data
  this->send_buffer_.insert(this->send_buffer_.end(),
                            this->accumulation_buffer_.begin(),
                            this->accumulation_buffer_.end());

  // Try to send UDP packet (non-blocking check for buffer space)
  ssize_t sent = sendto(this->sock_, this->send_buffer_.data(), this->send_buffer_.size(),
                        MSG_DONTWAIT,  // Non-blocking
                        (struct sockaddr *)&this->dest_addr_, sizeof(this->dest_addr_));

  if (sent > 0) {
    this->packets_sent_++;
    this->bytes_sent_ += sent;
    this->last_send_time_ = now;
  } else if (errno == ENOMEM || errno == EAGAIN || errno == EWOULDBLOCK) {
    // Buffer full - drop this batch silently (don't spam logs)
    this->packets_dropped_++;
  } else {
    // Other error - log it (but only occasionally)
    static uint32_t last_error_log = 0;
    if (now - last_error_log > 5000) {  // Log errors max once per 5 seconds
      ESP_LOGW(TAG, "Send failed: errno %d (dropped %d)", errno, this->packets_dropped_);
      last_error_log = now;
    }
    this->packets_dropped_++;
  }

  // Reset accumulation for next batch
  this->accumulation_buffer_.clear();
  this->batch_counter_ = 0;
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
  this->batch_counter_ = 0;
  this->accumulation_buffer_.clear();
  this->last_send_time_ = millis();
  this->reset_stats();
  this->last_stats_time_ = millis();

  ESP_LOGI(TAG, "Audio streaming started (batch=%d, interval=%dms)",
           this->batch_count_, this->send_interval_ms_);
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
  this->packets_dropped_ = 0;
}

}  // namespace audio_streamer
}  // namespace esphome
