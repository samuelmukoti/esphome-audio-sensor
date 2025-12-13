#pragma once

#include "esphome/core/component.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"

// ESP-IDF socket includes (works with both Arduino and ESP-IDF frameworks)
#include "lwip/sockets.h"
#include "lwip/netdb.h"

namespace esphome {
namespace detection_receiver {

/**
 * Component that receives beep detection results from an external server.
 *
 * Protocol: Server sends UDP packets with format:
 *   [1 byte: detected (0x00 or 0x01)] + [4 bytes: confidence (float, little-endian)]
 *
 * Architecture:
 *   ESP32 --UDP audio--> Server (NN) --UDP detection--> ESP32 --ESPHome API--> Home Assistant
 */
class DetectionReceiverComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Configuration setters
  void set_port(uint16_t port) { this->port_ = port; }

  // Sensor setters
  void set_binary_sensor(binary_sensor::BinarySensor *sensor) { this->binary_sensor_ = sensor; }
  void set_confidence_sensor(sensor::Sensor *sensor) { this->confidence_sensor_ = sensor; }
  void set_detection_count_sensor(sensor::Sensor *sensor) { this->detection_count_sensor_ = sensor; }

  // Runtime state
  uint32_t get_detection_count() const { return this->detection_count_; }
  void reset_detection_count();

 protected:
  void process_packet(uint8_t *data, size_t len);
  bool init_socket();
  void close_socket();

  // Components
  binary_sensor::BinarySensor *binary_sensor_{nullptr};
  sensor::Sensor *confidence_sensor_{nullptr};
  sensor::Sensor *detection_count_sensor_{nullptr};

  // Configuration
  uint16_t port_{5001};

  // UDP socket (ESP-IDF lwip)
  int sock_{-1};
  bool socket_ready_{false};

  // State
  uint32_t detection_count_{0};
  bool last_detection_state_{false};
  uint32_t last_packet_time_{0};

  // Auto-clear detection after timeout (no packet received)
  static const uint32_t DETECTION_TIMEOUT_MS = 2000;
};

}  // namespace detection_receiver
}  // namespace esphome
