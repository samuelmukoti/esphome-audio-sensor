#pragma once

#include "esphome/core/component.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/microphone/microphone.h"
#include <vector>
#include <cmath>

namespace esphome {
namespace beep_detector {

enum DetectionState {
  IDLE,
  DETECTING,
  CONFIRMED,
  COOLDOWN
};

class BeepDetectorComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::DATA; }

  // Configuration setters
  void set_microphone(microphone::Microphone *mic) { this->microphone_ = mic; }
  void set_binary_sensor(binary_sensor::BinarySensor *sensor) { this->binary_sensor_ = sensor; }
  void set_energy_sensor(sensor::Sensor *sensor) { this->energy_sensor_ = sensor; }
  void set_rms_sensor(sensor::Sensor *sensor) { this->rms_sensor_ = sensor; }
  void set_detection_count_sensor(sensor::Sensor *sensor) { this->detection_count_sensor_ = sensor; }

  void set_target_frequency(float freq) { this->target_frequency_ = freq; }
  void set_sample_rate(uint32_t rate) { this->sample_rate_ = rate; }
  void set_window_size_ms(uint32_t ms) { this->window_size_ms_ = ms; }
  void set_energy_threshold(float threshold) { this->energy_threshold_ = threshold; }
  void set_rms_threshold(float threshold) { this->rms_threshold_ = threshold; }
  void set_min_duration_ms(uint32_t ms) { this->min_duration_ms_ = ms; }
  void set_max_duration_ms(uint32_t ms) { this->max_duration_ms_ = ms; }
  void set_cooldown_ms(uint32_t ms) { this->cooldown_ms_ = ms; }
  void set_debounce_count(uint8_t count) { this->debounce_count_ = count; }

 protected:
  // Core processing
  void process_audio_data();
  float calculate_goertzel(const int16_t *samples, size_t count);
  float calculate_rms(const int16_t *samples, size_t count);
  bool detect_beep(float energy, float rms);
  void update_state_machine();

  // Components
  microphone::Microphone *microphone_{nullptr};
  binary_sensor::BinarySensor *binary_sensor_{nullptr};
  sensor::Sensor *energy_sensor_{nullptr};
  sensor::Sensor *rms_sensor_{nullptr};
  sensor::Sensor *detection_count_sensor_{nullptr};

  // Configuration parameters
  float target_frequency_{2615.0f};        // Hz - beep frequency
  uint32_t sample_rate_{16000};            // Hz
  uint32_t window_size_ms_{100};           // ms - analysis window
  float energy_threshold_{100.0f};         // Goertzel energy threshold
  float rms_threshold_{0.0069f};           // RMS amplitude threshold
  uint32_t min_duration_ms_{40};           // ms - minimum beep duration
  uint32_t max_duration_ms_{100};          // ms - maximum beep duration
  uint32_t cooldown_ms_{200};              // ms - cooldown after detection
  uint8_t debounce_count_{2};              // consecutive detections required

  // Goertzel algorithm coefficients
  float coeff_{0.0f};
  float sin_val_{0.0f};
  float cos_val_{0.0f};

  // State tracking
  DetectionState state_{IDLE};
  uint32_t detection_start_time_{0};
  uint32_t last_detection_time_{0};
  uint32_t cooldown_start_time_{0};
  uint8_t consecutive_detections_{0};
  uint32_t total_detections_{0};

  // Audio buffer
  std::vector<int16_t> audio_buffer_;
  size_t samples_per_window_{0};

  // Performance tracking
  uint32_t last_update_time_{0};
  uint32_t update_interval_ms_{50};  // Update sensors every 50ms
};

}  // namespace beep_detector
}  // namespace esphome
