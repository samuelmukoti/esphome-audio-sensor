#pragma once

#include "esphome/core/component.h"
#include "esphome/core/hal.h"
#include "esphome/components/microphone/microphone.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/sensor/sensor.h"

#include <vector>
#include <cmath>

namespace esphome {
namespace beep_detector_nn {

// MFCC extraction parameters
static const int N_MFCC = 20;
static const int N_FFT = 512;
static const int HOP_LENGTH = 160;  // 10ms at 16kHz
static const int N_MELS = 40;

// Model input shape
static const int MODEL_INPUT_FRAMES = 50;
static const int MODEL_INPUT_FEATURES = 20;

class BeepDetectorNNComponent : public Component {
 public:
  void setup() override;
  void loop() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Configuration setters
  void set_microphone(microphone::Microphone *mic) { this->microphone_ = mic; }
  void set_sample_rate(uint32_t rate) { this->sample_rate_ = rate; }
  void set_confidence_threshold(float threshold) { this->confidence_threshold_ = threshold; }
  void set_window_size_ms(uint32_t ms) { this->window_size_ms_ = ms; }
  void set_debounce_count(uint8_t count) { this->debounce_count_ = count; }

  // Sensor setters
  void set_binary_sensor(binary_sensor::BinarySensor *sensor) { this->binary_sensor_ = sensor; }
  void set_confidence_sensor(sensor::Sensor *sensor) { this->confidence_sensor_ = sensor; }
  void set_detection_count_sensor(sensor::Sensor *sensor) { this->detection_count_sensor_ = sensor; }

  // Runtime control
  void set_enabled(bool enabled) { this->enabled_ = enabled; }
  bool is_enabled() const { return this->enabled_; }
  void reset_detection_count();

 protected:
  // Audio processing
  void process_audio(const std::vector<uint8_t> &data);

  // MFCC extraction (simplified)
  void extract_mfcc(const int16_t *audio, int num_samples, float *mfcc_output);
  void compute_mel_spectrogram_frame(const float *frame, float *mel_output);

  // Neural network inference (manual implementation)
  float run_inference(const float *mfcc_input);

  // NN layer operations
  void conv1d_bn_relu(const float *input, float *output, int in_len, int in_ch, int out_ch,
                      const float *kernel, const float *bias,
                      const float *bn_gamma, const float *bn_beta,
                      const float *bn_mean, const float *bn_var);
  void maxpool1d(const float *input, float *output, int in_len, int channels);
  void global_avg_pool1d(const float *input, float *output, int in_len, int channels);
  void dense_relu(const float *input, float *output, int in_size, int out_size,
                  const float *kernel, const float *bias);
  void dense_sigmoid(const float *input, float *output, int in_size, int out_size,
                     const float *kernel, const float *bias);

  // FFT helper
  void compute_magnitude_spectrum(const float *signal, float *magnitude, int n);

  // Pre-computed mel filterbank
  void initialize_mel_filterbank();
  float hz_to_mel(float hz);
  float mel_to_hz(float mel);

  // Components
  microphone::Microphone *microphone_{nullptr};
  binary_sensor::BinarySensor *binary_sensor_{nullptr};
  sensor::Sensor *confidence_sensor_{nullptr};
  sensor::Sensor *detection_count_sensor_{nullptr};

  // Configuration
  uint32_t sample_rate_{16000};
  float confidence_threshold_{0.7f};
  uint32_t window_size_ms_{500};
  uint8_t debounce_count_{2};

  // State
  bool enabled_{true};
  bool initialized_{false};
  uint32_t detection_count_{0};
  uint8_t consecutive_detections_{0};
  bool last_detection_state_{false};

  // Audio buffer
  std::vector<int16_t> audio_buffer_;
  uint32_t window_samples_{0};
  float dc_offset_{0.0f};

  // Mel filterbank (pre-computed)
  std::vector<std::vector<float>> mel_filterbank_;

  // Intermediate buffers for inference
  std::vector<float> mfcc_buffer_;
  std::vector<float> layer_buffer_1_;
  std::vector<float> layer_buffer_2_;

  // Pre-allocated MFCC computation buffers (avoid stack allocation)
  std::vector<float> fft_frame_;
  std::vector<float> fft_magnitude_;
  std::vector<float> mel_energies_;

  // Timing
  uint32_t last_inference_time_{0};
  static const uint32_t INFERENCE_INTERVAL_MS = 250;
};

}  // namespace beep_detector_nn
}  // namespace esphome
