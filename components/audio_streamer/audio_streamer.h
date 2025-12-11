#pragma once

#include "esphome/core/component.h"
#include "esphome/components/microphone/microphone.h"
#include <WiFiUdp.h>
#include <vector>
#include <string>

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

  // Components
  microphone::Microphone *microphone_{nullptr};
  WiFiUDP udp_;

  // Configuration
  std::string target_ip_;
  uint16_t target_port_{5000};
  uint32_t sample_rate_{16000};
  bool enabled_{false};
  uint16_t chunk_size_{512};

  // State
  bool streaming_active_{false};
  uint32_t sequence_number_{0};
  uint32_t packets_sent_{0};
  uint32_t bytes_sent_{0};

  // Buffer for accumulating samples
  std::vector<uint8_t> send_buffer_;

  // Timing
  uint32_t last_stats_time_{0};
  static const uint32_t STATS_INTERVAL_MS = 5000;
};

}  // namespace audio_streamer
}  // namespace esphome
