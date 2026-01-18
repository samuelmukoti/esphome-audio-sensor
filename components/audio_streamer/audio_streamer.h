#pragma once

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"

#include <vector>
#include <string>

// ESP-IDF socket includes
#include "lwip/sockets.h"
#include "lwip/netdb.h"

namespace esphome {
namespace audio_streamer {

class AudioStreamerComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Configuration setters
  void set_microphone(microphone::Microphone *mic) { this->microphone_ = mic; }
  void set_target_ip(const std::string &ip) { this->target_ip_ = ip; }
  void set_target_port(uint16_t port) { this->target_port_ = port; }
  void set_sample_rate(uint32_t rate) { this->sample_rate_ = rate; }
  void set_enabled(bool enabled) { this->enabled_ = enabled; }
  void set_chunk_size(uint16_t size) { this->chunk_size_ = size; }
  void set_batch_count(uint8_t count) { this->batch_count_ = count; }  // How many mic reads to batch
  void set_send_interval_ms(uint32_t ms) { this->send_interval_ms_ = ms; }  // Min interval between sends

  // Runtime control (callable from Home Assistant)
  void start_streaming();
  void stop_streaming();
  bool is_streaming() const { return this->streaming_active_; }

  // Statistics
  uint32_t get_packets_sent() const { return this->packets_sent_; }
  uint32_t get_bytes_sent() const { return this->bytes_sent_; }
  void reset_stats();

 protected:
  void stream_audio_data(const std::vector<uint8_t> &data);
  bool init_socket();
  void close_socket();

  // Components
  microphone::Microphone *microphone_{nullptr};

  // Socket
  int sock_{-1};
  struct sockaddr_in dest_addr_{};

  // Configuration
  std::string target_ip_;
  uint16_t target_port_{5000};
  uint32_t sample_rate_{16000};
  bool enabled_{false};
  uint16_t chunk_size_{512};
  uint8_t batch_count_{4};  // Batch 4 mic reads (~64ms) before sending
  uint32_t send_interval_ms_{50};  // Min 50ms between sends = max 20 pkt/s

  // State
  bool streaming_active_{false};
  uint32_t sequence_number_{0};
  uint32_t packets_sent_{0};
  uint32_t bytes_sent_{0};
  uint32_t packets_dropped_{0};  // Dropped due to buffer full
  uint8_t batch_counter_{0};  // Current batch accumulation count

  // Buffer for accumulating samples
  std::vector<uint8_t> send_buffer_;
  std::vector<uint8_t> accumulation_buffer_;  // Buffer for batching multiple reads

  // Timing
  uint32_t last_stats_time_{0};
  uint32_t last_send_time_{0};  // For send interval throttling
  static const uint32_t STATS_INTERVAL_MS = 60000;  // Log stats every 60s instead of 5s
};

}  // namespace audio_streamer
}  // namespace esphome
