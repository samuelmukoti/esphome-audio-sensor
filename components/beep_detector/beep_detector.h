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

  // Configuration setters (used at setup time)
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

  // Runtime setters (can be called from HA to tune parameters)
  void set_energy_threshold_runtime(float threshold);
  void set_rms_threshold_runtime(float threshold);
  void set_target_frequency_runtime(float freq);
  void set_enabled(bool enabled);
  void reset_detection_count();

  // Getters (for HA number entities to read current values)
  float get_energy_threshold() const { return this->energy_threshold_; }
  float get_rms_threshold() const { return this->rms_threshold_; }
  float get_target_frequency() const { return this->target_frequency_; }
  uint32_t get_min_duration_ms() const { return this->min_duration_ms_; }
  uint32_t get_max_duration_ms() const { return this->max_duration_ms_; }
  uint32_t get_cooldown_ms() const { return this->cooldown_ms_; }
  uint8_t get_debounce_count() const { return this->debounce_count_; }
  bool is_enabled() const { return this->enabled_; }
  uint32_t get_total_detections() const { return this->total_detections_; }

 protected:
  // Core processing
  void process_audio_data();
  float calculate_goertzel(const int16_t *samples, size_t count);
  float calculate_rms(const int16_t *samples, size_t count);
  bool detect_beep(float energy, float rms);
  void update_state_machine();
  void recalculate_coefficients();

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
  bool enabled_{true};

  // Audio buffer
  std::vector<int16_t> audio_buffer_;
  size_t samples_per_window_{0};

  // Performance tracking
  uint32_t last_update_time_{0};
  uint32_t update_interval_ms_{50};  // Update sensors every 50ms
};

}  // namespace beep_detector
}  // namespace esphome
