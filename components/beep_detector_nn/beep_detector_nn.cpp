#include "beep_detector_nn.h"
#include "model_weights.h"
#include "esphome/core/log.h"

#include <cstring>
#include <algorithm>

namespace esphome {
namespace beep_detector_nn {

static const char *TAG = "beep_detector_nn";

void BeepDetectorNNComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Neural Network Beep Detector...");

  // Calculate window size in samples
  this->window_samples_ = this->sample_rate_ * this->window_size_ms_ / 1000;
  ESP_LOGCONFIG(TAG, "  Sample rate: %d Hz", this->sample_rate_);
  ESP_LOGCONFIG(TAG, "  Window size: %d ms (%d samples)", this->window_size_ms_, this->window_samples_);
  ESP_LOGCONFIG(TAG, "  Confidence threshold: %.2f", this->confidence_threshold_);
  ESP_LOGCONFIG(TAG, "  Debounce count: %d", this->debounce_count_);

  // Reserve buffer space
  this->audio_buffer_.reserve(this->window_samples_ * 2);

  // Initialize MFCC buffer (50 frames x 20 features)
  this->mfcc_buffer_.resize(MODEL_INPUT_FRAMES * MODEL_INPUT_FEATURES, 0.0f);

  // Initialize intermediate layer buffers
  // After conv1: 50 x 8, after pool: 25 x 8
  // After conv2: 25 x 16, after pool: 12 x 16
  // After conv3: 12 x 32, after GAP: 32
  this->layer_buffer_1_.resize(50 * 32, 0.0f);  // Largest needed
  this->layer_buffer_2_.resize(50 * 32, 0.0f);

  // Initialize mel filterbank
  this->initialize_mel_filterbank();

  // Register audio callback with microphone
  if (this->microphone_ != nullptr) {
    this->microphone_->add_data_callback([this](const std::vector<uint8_t> &data) {
      if (this->enabled_) {
        this->process_audio(data);
      }
    });
    this->microphone_->start();
    ESP_LOGI(TAG, "Microphone started");
  } else {
    ESP_LOGE(TAG, "Microphone is NULL!");
    return;
  }

  this->initialized_ = true;
  this->last_inference_time_ = millis();
  ESP_LOGI(TAG, "Neural Network Beep Detector initialized");
}

void BeepDetectorNNComponent::loop() {
  if (!this->initialized_ || !this->enabled_) {
    return;
  }

  uint32_t now = millis();

  // Run inference at regular intervals when we have enough data
  if (now - this->last_inference_time_ >= INFERENCE_INTERVAL_MS &&
      this->audio_buffer_.size() >= this->window_samples_) {

    // Extract MFCC features from the audio buffer
    this->extract_mfcc(this->audio_buffer_.data(),
                       std::min((int)this->audio_buffer_.size(), (int)this->window_samples_),
                       this->mfcc_buffer_.data());

    // Run neural network inference
    float confidence = this->run_inference(this->mfcc_buffer_.data());

    // Update confidence sensor
    if (this->confidence_sensor_ != nullptr) {
      this->confidence_sensor_->publish_state(confidence);
    }

    // Detection logic with debouncing
    bool is_beep = confidence > this->confidence_threshold_;

    if (is_beep) {
      this->consecutive_detections_++;
      ESP_LOGD(TAG, "Beep candidate: confidence=%.3f, consecutive=%d",
               confidence, this->consecutive_detections_);
    } else {
      this->consecutive_detections_ = 0;
    }

    // Confirmed detection after debounce
    bool confirmed = this->consecutive_detections_ >= this->debounce_count_;

    if (confirmed && !this->last_detection_state_) {
      // New detection
      this->detection_count_++;
      ESP_LOGI(TAG, "BEEP DETECTED! confidence=%.3f, total=%d",
               confidence, this->detection_count_);

      if (this->binary_sensor_ != nullptr) {
        this->binary_sensor_->publish_state(true);
      }
      if (this->detection_count_sensor_ != nullptr) {
        this->detection_count_sensor_->publish_state(this->detection_count_);
      }
    } else if (!confirmed && this->last_detection_state_) {
      // Detection ended
      if (this->binary_sensor_ != nullptr) {
        this->binary_sensor_->publish_state(false);
      }
    }

    this->last_detection_state_ = confirmed;
    this->last_inference_time_ = now;

    // Trim audio buffer to prevent memory growth
    if (this->audio_buffer_.size() > this->window_samples_ * 2) {
      this->audio_buffer_.erase(
          this->audio_buffer_.begin(),
          this->audio_buffer_.begin() + this->audio_buffer_.size() - this->window_samples_);
    }
  }
}

void BeepDetectorNNComponent::process_audio(const std::vector<uint8_t> &data) {
  // Convert uint8_t bytes to int16_t samples with DC offset removal
  for (size_t i = 0; i + 1 < data.size(); i += 2) {
    int16_t raw_sample = (int16_t)((data[i + 1] << 8) | data[i]);

    // Update DC offset estimate (exponential moving average)
    this->dc_offset_ = this->dc_offset_ * 0.999f + (float)raw_sample * 0.001f;

    // Remove DC offset
    int16_t sample = raw_sample - (int16_t)this->dc_offset_;
    this->audio_buffer_.push_back(sample);
  }
}

void BeepDetectorNNComponent::extract_mfcc(const int16_t *audio, int num_samples, float *mfcc_output) {
  // Simplified MFCC extraction
  // We compute a rough approximation suitable for our trained model

  const int frame_length = N_FFT;
  const int hop_length = HOP_LENGTH;
  const int num_frames = MODEL_INPUT_FRAMES;

  // Working buffers
  std::vector<float> frame(frame_length, 0.0f);
  std::vector<float> magnitude(frame_length / 2 + 1, 0.0f);
  std::vector<float> mel_energies(N_MELS, 0.0f);

  // Process each frame
  for (int f = 0; f < num_frames; f++) {
    int start = f * hop_length;

    // Extract and window the frame
    for (int i = 0; i < frame_length; i++) {
      int idx = start + i;
      if (idx < num_samples) {
        // Apply Hann window
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (frame_length - 1)));
        frame[i] = (float)audio[idx] / 32768.0f * window;
      } else {
        frame[i] = 0.0f;
      }
    }

    // Compute magnitude spectrum (simplified FFT approximation)
    this->compute_magnitude_spectrum(frame.data(), magnitude.data(), frame_length);

    // Apply mel filterbank
    for (int m = 0; m < N_MELS; m++) {
      mel_energies[m] = 0.0f;
      for (size_t k = 0; k < this->mel_filterbank_[m].size(); k++) {
        if (k < magnitude.size()) {
          mel_energies[m] += this->mel_filterbank_[m][k] * magnitude[k];
        }
      }
      // Log compression
      mel_energies[m] = logf(mel_energies[m] + 1e-10f);
    }

    // DCT to get MFCCs (simplified - just use first N_MFCC mel coefficients)
    for (int c = 0; c < N_MFCC; c++) {
      float sum = 0.0f;
      for (int m = 0; m < N_MELS; m++) {
        sum += mel_energies[m] * cosf(M_PI * c * (m + 0.5f) / N_MELS);
      }
      mfcc_output[f * N_MFCC + c] = sum * sqrtf(2.0f / N_MELS);
    }
  }
}

void BeepDetectorNNComponent::compute_magnitude_spectrum(const float *signal, float *magnitude, int n) {
  // Simplified DFT for small n (not optimized, but works)
  // For ESP32, consider using ESP-DSP library for better performance
  int half_n = n / 2 + 1;

  for (int k = 0; k < half_n; k++) {
    float real = 0.0f;
    float imag = 0.0f;
    for (int i = 0; i < n; i++) {
      float angle = -2.0f * M_PI * k * i / n;
      real += signal[i] * cosf(angle);
      imag += signal[i] * sinf(angle);
    }
    magnitude[k] = sqrtf(real * real + imag * imag);
  }
}

void BeepDetectorNNComponent::initialize_mel_filterbank() {
  // Create mel filterbank
  this->mel_filterbank_.resize(N_MELS);

  float mel_min = this->hz_to_mel(0.0f);
  float mel_max = this->hz_to_mel(this->sample_rate_ / 2.0f);

  // Mel points
  std::vector<float> mel_points(N_MELS + 2);
  for (int i = 0; i < N_MELS + 2; i++) {
    mel_points[i] = mel_min + i * (mel_max - mel_min) / (N_MELS + 1);
  }

  // Convert to Hz and then to FFT bins
  std::vector<int> bin_points(N_MELS + 2);
  for (int i = 0; i < N_MELS + 2; i++) {
    float hz = this->mel_to_hz(mel_points[i]);
    bin_points[i] = (int)floorf((N_FFT + 1) * hz / this->sample_rate_);
  }

  // Create triangular filters
  int num_bins = N_FFT / 2 + 1;
  for (int m = 0; m < N_MELS; m++) {
    this->mel_filterbank_[m].resize(num_bins, 0.0f);

    for (int k = bin_points[m]; k < bin_points[m + 1]; k++) {
      if (k < num_bins) {
        this->mel_filterbank_[m][k] = (float)(k - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
      }
    }
    for (int k = bin_points[m + 1]; k < bin_points[m + 2]; k++) {
      if (k < num_bins) {
        this->mel_filterbank_[m][k] = (float)(bin_points[m + 2] - k) / (bin_points[m + 2] - bin_points[m + 1]);
      }
    }
  }

  ESP_LOGD(TAG, "Mel filterbank initialized with %d filters", N_MELS);
}

float BeepDetectorNNComponent::hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float BeepDetectorNNComponent::mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

float BeepDetectorNNComponent::run_inference(const float *mfcc_input) {
  // Manual neural network forward pass
  // Architecture: Conv1D(8) -> BN -> ReLU -> Pool ->
  //               Conv1D(16) -> BN -> ReLU -> Pool ->
  //               Conv1D(32) -> BN -> ReLU -> GAP ->
  //               Dense(16) -> ReLU -> Dense(8) -> ReLU -> Dense(1) -> Sigmoid

  float *buf1 = this->layer_buffer_1_.data();
  float *buf2 = this->layer_buffer_2_.data();

  // Layer 1: Conv1D(8) + BatchNorm + ReLU
  // Input: (50, 20), Output: (50, 8)
  this->conv1d_bn_relu(mfcc_input, buf1, 50, 20, 8,
                       conv1d_kernel, conv1d_bias,
                       batchnormalization_gamma, batchnormalization_beta,
                       batchnormalization_mean, batchnormalization_var);

  // MaxPool1D: (50, 8) -> (25, 8)
  this->maxpool1d(buf1, buf2, 50, 8);

  // Layer 2: Conv1D(16) + BatchNorm + ReLU
  // Input: (25, 8), Output: (25, 16)
  this->conv1d_bn_relu(buf2, buf1, 25, 8, 16,
                       conv1d1_kernel, conv1d1_bias,
                       batchnormalization1_gamma, batchnormalization1_beta,
                       batchnormalization1_mean, batchnormalization1_var);

  // MaxPool1D: (25, 16) -> (12, 16)
  this->maxpool1d(buf1, buf2, 25, 16);

  // Layer 3: Conv1D(32) + BatchNorm + ReLU
  // Input: (12, 16), Output: (12, 32)
  this->conv1d_bn_relu(buf2, buf1, 12, 16, 32,
                       conv1d2_kernel, conv1d2_bias,
                       batchnormalization2_gamma, batchnormalization2_beta,
                       batchnormalization2_mean, batchnormalization2_var);

  // GlobalAveragePooling1D: (12, 32) -> (32,)
  this->global_avg_pool1d(buf1, buf2, 12, 32);

  // Dense(16) + ReLU
  this->dense_relu(buf2, buf1, 32, 16, dense_kernel, dense_bias);

  // Dense(8) + ReLU (skip dropout at inference)
  this->dense_relu(buf1, buf2, 16, 8, dense1_kernel, dense1_bias);

  // Dense(1) + Sigmoid
  float output;
  this->dense_sigmoid(buf2, &output, 8, 1, dense2_kernel, dense2_bias);

  return output;
}

void BeepDetectorNNComponent::conv1d_bn_relu(
    const float *input, float *output, int in_len, int in_ch, int out_ch,
    const float *kernel, const float *bias,
    const float *bn_gamma, const float *bn_beta,
    const float *bn_mean, const float *bn_var) {

  const int kernel_size = 3;
  const float epsilon = 1e-5f;

  // Conv1D with 'same' padding
  for (int t = 0; t < in_len; t++) {
    for (int oc = 0; oc < out_ch; oc++) {
      float sum = bias[oc];

      for (int k = 0; k < kernel_size; k++) {
        int ti = t + k - kernel_size / 2;  // Same padding
        if (ti >= 0 && ti < in_len) {
          for (int ic = 0; ic < in_ch; ic++) {
            // kernel shape: (kernel_size, in_ch, out_ch)
            int kernel_idx = k * in_ch * out_ch + ic * out_ch + oc;
            sum += input[ti * in_ch + ic] * kernel[kernel_idx];
          }
        }
      }

      // BatchNorm: gamma * (x - mean) / sqrt(var + eps) + beta
      float normalized = (sum - bn_mean[oc]) / sqrtf(bn_var[oc] + epsilon);
      float bn_out = bn_gamma[oc] * normalized + bn_beta[oc];

      // ReLU
      output[t * out_ch + oc] = bn_out > 0.0f ? bn_out : 0.0f;
    }
  }
}

void BeepDetectorNNComponent::maxpool1d(const float *input, float *output, int in_len, int channels) {
  int out_len = in_len / 2;
  for (int t = 0; t < out_len; t++) {
    for (int c = 0; c < channels; c++) {
      float max_val = input[(t * 2) * channels + c];
      float val2 = input[(t * 2 + 1) * channels + c];
      output[t * channels + c] = max_val > val2 ? max_val : val2;
    }
  }
}

void BeepDetectorNNComponent::global_avg_pool1d(const float *input, float *output, int in_len, int channels) {
  for (int c = 0; c < channels; c++) {
    float sum = 0.0f;
    for (int t = 0; t < in_len; t++) {
      sum += input[t * channels + c];
    }
    output[c] = sum / in_len;
  }
}

void BeepDetectorNNComponent::dense_relu(
    const float *input, float *output, int in_size, int out_size,
    const float *kernel, const float *bias) {

  for (int o = 0; o < out_size; o++) {
    float sum = bias[o];
    for (int i = 0; i < in_size; i++) {
      // kernel shape: (in_size, out_size)
      sum += input[i] * kernel[i * out_size + o];
    }
    // ReLU
    output[o] = sum > 0.0f ? sum : 0.0f;
  }
}

void BeepDetectorNNComponent::dense_sigmoid(
    const float *input, float *output, int in_size, int out_size,
    const float *kernel, const float *bias) {

  for (int o = 0; o < out_size; o++) {
    float sum = bias[o];
    for (int i = 0; i < in_size; i++) {
      sum += input[i] * kernel[i * out_size + o];
    }
    // Sigmoid
    output[o] = 1.0f / (1.0f + expf(-sum));
  }
}

void BeepDetectorNNComponent::reset_detection_count() {
  this->detection_count_ = 0;
  if (this->detection_count_sensor_ != nullptr) {
    this->detection_count_sensor_->publish_state(0);
  }
  ESP_LOGI(TAG, "Detection count reset");
}

}  // namespace beep_detector_nn
}  // namespace esphome
