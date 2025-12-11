#include "beep_detector.h"
#include "esphome/core/log.h"
#include "esphome/core/hal.h"

namespace esphome {
namespace beep_detector {

static const char *TAG = "beep_detector";

void BeepDetectorComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Beep Detector...");

  // Calculate samples per window
  this->samples_per_window_ = (this->sample_rate_ * this->window_size_ms_) / 1000;
  this->audio_buffer_.reserve(this->samples_per_window_ * 2);  // Extra space for overlap

  // Pre-calculate Goertzel coefficients for target frequency
  // k = (int)(0.5 + ((N * target_freq) / sample_rate))
  // coeff = 2 * cos(2 * PI * k / N)
  float k = 0.5f + ((float)this->samples_per_window_ * this->target_frequency_ / (float)this->sample_rate_);
  float omega = (2.0f * M_PI * k) / (float)this->samples_per_window_;
  this->coeff_ = 2.0f * cosf(omega);
  this->sin_val_ = sinf(omega);
  this->cos_val_ = cosf(omega);

  ESP_LOGCONFIG(TAG, "  Target Frequency: %.1f Hz", this->target_frequency_);
  ESP_LOGCONFIG(TAG, "  Sample Rate: %d Hz", this->sample_rate_);
  ESP_LOGCONFIG(TAG, "  Window Size: %d ms (%d samples)", this->window_size_ms_, this->samples_per_window_);
  ESP_LOGCONFIG(TAG, "  Energy Threshold: %.2f", this->energy_threshold_);
  ESP_LOGCONFIG(TAG, "  RMS Threshold: %.4f", this->rms_threshold_);
  ESP_LOGCONFIG(TAG, "  Duration Range: %d-%d ms", this->min_duration_ms_, this->max_duration_ms_);
  ESP_LOGCONFIG(TAG, "  Goertzel Coefficient: %.6f", this->coeff_);

  if (this->binary_sensor_ != nullptr) {
    this->binary_sensor_->publish_state(false);
  }

  // Register data callback with microphone
  if (this->microphone_ != nullptr) {
    this->microphone_->add_data_callback([this](const std::vector<uint8_t> &data) {
      // Debug: log callback activity
      static uint32_t callback_count = 0;
      static int16_t max_sample = -32768;
      static int16_t min_sample = 32767;
      static float dc_offset = 1400.0f;  // Initial estimate for PDM mic DC bias
      callback_count++;

      // Convert uint8_t bytes to int16_t samples (little-endian) with DC offset removal
      for (size_t i = 0; i + 1 < data.size(); i += 2) {
        int16_t raw_sample = (int16_t)((data[i + 1] << 8) | data[i]);

        // Update DC offset estimate using exponential moving average
        // Alpha = 0.001 means slow adaptation (filters out audio, keeps DC)
        dc_offset = dc_offset * 0.999f + (float)raw_sample * 0.001f;

        // Remove DC offset to get AC-coupled audio signal
        int16_t sample = raw_sample - (int16_t)dc_offset;
        this->audio_buffer_.push_back(sample);

        // Track min/max for debug (after DC removal)
        if (sample > max_sample) max_sample = sample;
        if (sample < min_sample) min_sample = sample;
      }

      // Log every 100 callbacks (~every second)
      if (callback_count % 100 == 0) {
        ESP_LOGD(TAG, "Audio callback #%d: buffer=%d, range=[%d, %d], dc_offset=%.0f",
                 callback_count, this->audio_buffer_.size(), min_sample, max_sample, dc_offset);
        // Reset min/max for next interval
        min_sample = 32767;
        max_sample = -32768;
      }
    });
    this->microphone_->start();
    ESP_LOGI(TAG, "Microphone started");
  } else {
    ESP_LOGE(TAG, "Microphone is NULL - cannot start audio capture!");
  }

  this->state_ = IDLE;
  this->last_update_time_ = millis();
}

void BeepDetectorComponent::loop() {
  if (this->microphone_ == nullptr || !this->enabled_) {
    return;
  }

  // Process audio data if available
  this->process_audio_data();

  // Update state machine
  this->update_state_machine();

  // Update sensors periodically (not every loop to reduce overhead)
  uint32_t now = millis();
  if (now - this->last_update_time_ >= this->update_interval_ms_) {
    if (this->detection_count_sensor_ != nullptr) {
      this->detection_count_sensor_->publish_state(this->total_detections_);
    }
    this->last_update_time_ = now;
  }
}

void BeepDetectorComponent::process_audio_data() {
  // Process when we have enough samples for a full window
  // Data is collected via callback registered in setup()
  if (this->audio_buffer_.size() >= this->samples_per_window_) {
    // Calculate Goertzel energy for target frequency
    float energy = this->calculate_goertzel(this->audio_buffer_.data(), this->samples_per_window_);

    // Calculate RMS for overall amplitude
    float rms = this->calculate_rms(this->audio_buffer_.data(), this->samples_per_window_);

    // Update diagnostic sensors
    if (this->energy_sensor_ != nullptr) {
      this->energy_sensor_->publish_state(energy);
    }
    if (this->rms_sensor_ != nullptr) {
      this->rms_sensor_->publish_state(rms);
    }

    // Detect beep based on criteria
    bool detected = this->detect_beep(energy, rms);

    // Log detection events
    if (detected) {
      ESP_LOGD(TAG, "Beep detected: energy=%.2f (thresh=%.2f), rms=%.4f (thresh=%.4f)",
               energy, this->energy_threshold_, rms, this->rms_threshold_);
    }

    // Clear buffer for next window (sliding window approach)
    // Keep last 25% of samples for overlap
    size_t overlap = this->samples_per_window_ / 4;
    if (this->audio_buffer_.size() > overlap) {
      this->audio_buffer_.erase(this->audio_buffer_.begin(),
                                this->audio_buffer_.end() - overlap);
    }
  }
}

float BeepDetectorComponent::calculate_goertzel(const int16_t *samples, size_t count) {
  // Goertzel algorithm implementation
  // This efficiently calculates the magnitude of a specific frequency component

  float q0 = 0.0f;
  float q1 = 0.0f;
  float q2 = 0.0f;

  // Process all samples
  for (size_t i = 0; i < count; i++) {
    // Normalize sample to [-1, 1] range
    float sample = (float)samples[i] / 32768.0f;

    q0 = this->coeff_ * q1 - q2 + sample;
    q2 = q1;
    q1 = q0;
  }

  // Calculate magnitude squared
  float real = q1 - q2 * this->cos_val_;
  float imag = q2 * this->sin_val_;
  float magnitude_squared = real * real + imag * imag;

  // Return normalized energy (scale by window size)
  return magnitude_squared / (float)count;
}

float BeepDetectorComponent::calculate_rms(const int16_t *samples, size_t count) {
  // Calculate Root Mean Square for amplitude detection
  float sum_squares = 0.0f;

  for (size_t i = 0; i < count; i++) {
    float sample = (float)samples[i] / 32768.0f;
    sum_squares += sample * sample;
  }

  return sqrtf(sum_squares / (float)count);
}

bool BeepDetectorComponent::detect_beep(float energy, float rms) {
  // Multi-criteria detection:
  // 1. Energy at target frequency exceeds threshold (Goertzel)
  // 2. Overall RMS amplitude exceeds threshold
  // 3. Duration validation happens in state machine

  bool frequency_match = (energy > this->energy_threshold_);
  bool amplitude_match = (rms > this->rms_threshold_);

  return frequency_match && amplitude_match;
}

void BeepDetectorComponent::update_state_machine() {
  uint32_t now = millis();

  switch (this->state_) {
    case IDLE:
      // Waiting for first detection
      if (this->consecutive_detections_ > 0) {
        this->state_ = DETECTING;
        this->detection_start_time_ = now;
        ESP_LOGD(TAG, "State: IDLE -> DETECTING");
      }
      break;

    case DETECTING:
      // Accumulating consecutive detections
      if (this->consecutive_detections_ >= this->debounce_count_) {
        uint32_t duration = now - this->detection_start_time_;

        // Validate duration
        if (duration >= this->min_duration_ms_ && duration <= this->max_duration_ms_) {
          this->state_ = CONFIRMED;
          this->total_detections_++;
          this->last_detection_time_ = now;

          ESP_LOGI(TAG, "Beep CONFIRMED! Duration: %d ms, Total: %d",
                   duration, this->total_detections_);

          if (this->binary_sensor_ != nullptr) {
            this->binary_sensor_->publish_state(true);
          }
        } else {
          ESP_LOGD(TAG, "Detection failed duration check: %d ms (range: %d-%d ms)",
                   duration, this->min_duration_ms_, this->max_duration_ms_);
          this->state_ = IDLE;
          this->consecutive_detections_ = 0;
        }
      } else if (this->consecutive_detections_ == 0) {
        // Lost detection before debounce threshold
        ESP_LOGD(TAG, "State: DETECTING -> IDLE (lost signal)");
        this->state_ = IDLE;
      } else if (now - this->detection_start_time_ > this->max_duration_ms_) {
        // Exceeded maximum duration
        ESP_LOGD(TAG, "State: DETECTING -> IDLE (exceeded max duration)");
        this->state_ = IDLE;
        this->consecutive_detections_ = 0;
      }
      break;

    case CONFIRMED:
      // Enter cooldown to prevent duplicate detections
      this->state_ = COOLDOWN;
      this->cooldown_start_time_ = now;
      this->consecutive_detections_ = 0;
      ESP_LOGD(TAG, "State: CONFIRMED -> COOLDOWN");
      break;

    case COOLDOWN:
      // Wait for cooldown period to expire
      if (now - this->cooldown_start_time_ >= this->cooldown_ms_) {
        this->state_ = IDLE;
        ESP_LOGD(TAG, "State: COOLDOWN -> IDLE");

        if (this->binary_sensor_ != nullptr) {
          this->binary_sensor_->publish_state(false);
        }
      }
      break;
  }
}

void BeepDetectorComponent::recalculate_coefficients() {
  // Recalculate Goertzel coefficients for new target frequency
  float k = 0.5f + ((float)this->samples_per_window_ * this->target_frequency_ / (float)this->sample_rate_);
  float omega = (2.0f * M_PI * k) / (float)this->samples_per_window_;
  this->coeff_ = 2.0f * cosf(omega);
  this->sin_val_ = sinf(omega);
  this->cos_val_ = cosf(omega);
  ESP_LOGI(TAG, "Recalculated coefficients for %.1f Hz (coeff=%.6f)", this->target_frequency_, this->coeff_);
}

void BeepDetectorComponent::set_energy_threshold_runtime(float threshold) {
  this->energy_threshold_ = threshold;
  ESP_LOGI(TAG, "Energy threshold set to %.4f", threshold);
}

void BeepDetectorComponent::set_rms_threshold_runtime(float threshold) {
  this->rms_threshold_ = threshold;
  ESP_LOGI(TAG, "RMS threshold set to %.6f", threshold);
}

void BeepDetectorComponent::set_target_frequency_runtime(float freq) {
  this->target_frequency_ = freq;
  this->recalculate_coefficients();
}

void BeepDetectorComponent::set_enabled(bool enabled) {
  this->enabled_ = enabled;
  ESP_LOGI(TAG, "Beep detection %s", enabled ? "ENABLED" : "DISABLED");
  if (!enabled && this->binary_sensor_ != nullptr) {
    this->binary_sensor_->publish_state(false);
  }
}

void BeepDetectorComponent::reset_detection_count() {
  this->total_detections_ = 0;
  if (this->detection_count_sensor_ != nullptr) {
    this->detection_count_sensor_->publish_state(0);
  }
  ESP_LOGI(TAG, "Detection count reset to 0");
}

}  // namespace beep_detector
}  // namespace esphome
