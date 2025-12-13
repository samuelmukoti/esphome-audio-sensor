#include "beep_detector_nn.h"
#include "model_weights.h"
#include "esphome/core/log.h"
#include "esphome/core/application.h"

#include <cstring>
#include <algorithm>

namespace esphome {
namespace beep_detector_nn {

static const char *TAG = "beep_detector_nn";

void BeepDetectorNNComponent::setup() {
  ESP_LOGCONFIG(TAG, "Setting up Neural Network Beep Detector (v2 - memory optimized)...");

#ifdef ESP32
  ESP_LOGD(TAG, "  Free heap before allocation: %d bytes", ESP.getFreeHeap());
#endif

  // Calculate window size in samples
  this->window_samples_ = this->sample_rate_ * this->window_size_ms_ / 1000;
  ESP_LOGCONFIG(TAG, "  Sample rate: %d Hz", this->sample_rate_);
  ESP_LOGCONFIG(TAG, "  Window size: %d ms (%d samples)", this->window_size_ms_, this->window_samples_);
  ESP_LOGCONFIG(TAG, "  Confidence threshold: %.2f", this->confidence_threshold_);
  ESP_LOGCONFIG(TAG, "  Debounce count: %d", this->debounce_count_);
  ESP_LOGCONFIG(TAG, "  Model: 2-layer CNN (8-8 filters), %d frames", MODEL_INPUT_FRAMES);

  // Pre-allocate audio buffers immediately (not lazy reserve!)
  // This ensures we get the memory upfront rather than failing during callback
  this->audio_buffer_.resize(this->window_samples_, 0);  // Ring buffer
  this->audio_linear_.resize(this->window_samples_, 0);  // Linear copy for MFCC
  this->audio_write_pos_ = 0;
  ESP_LOGD(TAG, "  Audio buffers: %d bytes (ring + linear)", this->window_samples_ * 4);

  // Initialize MFCC buffer (25 frames x 20 features = 500 floats = 2KB)
  this->mfcc_buffer_.resize(MODEL_INPUT_FRAMES * MODEL_INPUT_FEATURES, 0.0f);
  ESP_LOGD(TAG, "  MFCC buffer: %d bytes", MODEL_INPUT_FRAMES * MODEL_INPUT_FEATURES * 4);

  // Initialize intermediate layer buffers (much smaller for 2-layer model)
  // After conv1: 25 x 8, after pool: 12 x 8
  // After conv2: 12 x 8, after GAP: 8
  // Max needed: 25 * 8 = 200 floats = 800 bytes each
  this->layer_buffer_1_.resize(MODEL_INPUT_FRAMES * CONV1_FILTERS, 0.0f);
  this->layer_buffer_2_.resize(MODEL_INPUT_FRAMES * CONV2_FILTERS, 0.0f);
  ESP_LOGD(TAG, "  Layer buffers: %d bytes each", MODEL_INPUT_FRAMES * CONV1_FILTERS * 4);

  // Pre-allocate MFCC computation buffers
  this->fft_frame_.resize(N_FFT, 0.0f);
  this->fft_magnitude_.resize(N_FFT_BINS, 0.0f);
  this->mel_energies_.resize(N_MELS, 0.0f);

  // Initialize FFT work buffer for built-in Cooley-Tukey FFT
  this->fft_work_buffer_.resize(N_FFT * 2, 0.0f);  // Complex pairs (real, imag)
  this->fft_initialized_ = true;
  ESP_LOGD(TAG, "  Built-in FFT initialized for N=%d", N_FFT);

  // Initialize mel filterbank (flat array instead of nested vectors)
  this->initialize_mel_filterbank();

#ifdef ESP32
  ESP_LOGD(TAG, "  Free heap after allocation: %d bytes", ESP.getFreeHeap());
#endif

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
  ESP_LOGI(TAG, "Neural Network Beep Detector initialized successfully");
}

void BeepDetectorNNComponent::loop() {
  if (!this->initialized_ || !this->enabled_) {
    return;
  }

  uint32_t now = millis();

  // Run inference at regular intervals when we have enough data
  if (now - this->last_inference_time_ >= INFERENCE_INTERVAL_MS &&
      this->audio_write_pos_ >= this->window_samples_) {

    // Feed watchdog before heavy computation
    App.feed_wdt();

    // Copy ring buffer to linear buffer in correct temporal order
    // The ring buffer wraps around, so we need to unwrap it:
    // - Oldest sample is at (audio_write_pos_ % window_samples_)
    // - Copy from that position to end, then from 0 to that position
    uint32_t start_pos = this->audio_write_pos_ % this->window_samples_;
    uint32_t first_chunk = this->window_samples_ - start_pos;

    // Copy from start_pos to end of buffer
    std::memcpy(this->audio_linear_.data(),
                this->audio_buffer_.data() + start_pos,
                first_chunk * sizeof(int16_t));

    // Copy from beginning to start_pos (if any)
    if (start_pos > 0) {
      std::memcpy(this->audio_linear_.data() + first_chunk,
                  this->audio_buffer_.data(),
                  start_pos * sizeof(int16_t));
    }

    // Extract MFCC features from the linearized audio buffer
    this->extract_mfcc(this->audio_linear_.data(),
                       this->window_samples_,
                       this->mfcc_buffer_.data());

    // Feed watchdog between MFCC and inference
    App.feed_wdt();

    // Run neural network inference
    float confidence = this->run_inference(this->mfcc_buffer_.data());

    // Feed watchdog after inference
    App.feed_wdt();

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

    // No need to trim buffer - using fixed-size circular buffer now
  }
}

void BeepDetectorNNComponent::process_audio(const std::vector<uint8_t> &data) {
  // Convert uint8_t bytes to int16_t samples with DC offset removal
  // Using TRUE ring buffer - just write at position and increment
  // No memmove! This is O(1) per sample instead of O(n)
  for (size_t i = 0; i + 1 < data.size(); i += 2) {
    int16_t raw_sample = (int16_t)((data[i + 1] << 8) | data[i]);

    // Update DC offset estimate (exponential moving average)
    this->dc_offset_ = this->dc_offset_ * 0.999f + (float)raw_sample * 0.001f;

    // Remove DC offset
    int16_t sample = raw_sample - (int16_t)this->dc_offset_;

    // Write directly to ring buffer position - O(1) operation!
    this->audio_buffer_[this->audio_write_pos_ % this->window_samples_] = sample;
    this->audio_write_pos_++;
  }
}

void BeepDetectorNNComponent::extract_mfcc(const int16_t *audio, int num_samples, float *mfcc_output) {
  // Simplified MFCC extraction using pre-allocated buffers
  // We compute a rough approximation suitable for our trained model

  const int frame_length = N_FFT;
  const int hop_length = HOP_LENGTH;
  const int num_frames = MODEL_INPUT_FRAMES;

  // Use pre-allocated buffers (avoid stack overflow)
  float *frame = this->fft_frame_.data();
  float *magnitude = this->fft_magnitude_.data();
  float *mel_energies = this->mel_energies_.data();

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

    // Compute magnitude spectrum (FFT - fast!)
    this->compute_magnitude_spectrum(frame, magnitude, frame_length);

    // Feed watchdog every frame (no delay needed - FFT is fast)
    App.feed_wdt();

    // Apply mel filterbank (using flat array layout)
    for (int m = 0; m < N_MELS; m++) {
      mel_energies[m] = 0.0f;
      int filter_offset = m * N_FFT_BINS;
      for (int k = 0; k < N_FFT_BINS; k++) {
        mel_energies[m] += this->mel_filterbank_[filter_offset + k] * magnitude[k];
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
  // Built-in Cooley-Tukey radix-2 FFT - O(n log n) instead of O(n²)
  // No external library needed, ~100x faster than naive DFT

  if (!this->fft_initialized_) {
    ESP_LOGE(TAG, "FFT not initialized!");
    return;
  }

  float *work = this->fft_work_buffer_.data();

  // Copy signal to work buffer as complex (real, imag pairs)
  for (int i = 0; i < n; i++) {
    work[i * 2] = signal[i];      // Real part
    work[i * 2 + 1] = 0.0f;       // Imaginary part = 0
  }

  // Bit-reversal permutation
  int j = 0;
  for (int i = 0; i < n - 1; i++) {
    if (i < j) {
      // Swap complex values at i and j
      float temp_r = work[i * 2];
      float temp_i = work[i * 2 + 1];
      work[i * 2] = work[j * 2];
      work[i * 2 + 1] = work[j * 2 + 1];
      work[j * 2] = temp_r;
      work[j * 2 + 1] = temp_i;
    }
    int k = n / 2;
    while (k <= j) {
      j -= k;
      k /= 2;
    }
    j += k;
  }

  // Cooley-Tukey butterfly operations
  for (int stage = 1; stage < n; stage *= 2) {
    float angle_step = -M_PI / stage;
    float w_r = 1.0f;
    float w_i = 0.0f;
    float cos_step = cosf(angle_step);
    float sin_step = sinf(angle_step);

    for (int group = 0; group < stage; group++) {
      for (int pair = group; pair < n; pair += stage * 2) {
        int match = pair + stage;

        // Butterfly: (a, b) -> (a + w*b, a - w*b)
        float t_r = w_r * work[match * 2] - w_i * work[match * 2 + 1];
        float t_i = w_r * work[match * 2 + 1] + w_i * work[match * 2];

        work[match * 2] = work[pair * 2] - t_r;
        work[match * 2 + 1] = work[pair * 2 + 1] - t_i;
        work[pair * 2] += t_r;
        work[pair * 2 + 1] += t_i;
      }
      // Rotate twiddle factor
      float temp = w_r * cos_step - w_i * sin_step;
      w_i = w_r * sin_step + w_i * cos_step;
      w_r = temp;
    }
  }

  // Extract magnitude for positive frequencies only
  int half_n = n / 2 + 1;
  for (int k = 0; k < half_n; k++) {
    float real = work[k * 2];
    float imag = work[k * 2 + 1];
    magnitude[k] = sqrtf(real * real + imag * imag);
  }
}

void BeepDetectorNNComponent::initialize_mel_filterbank() {
  // Create mel filterbank as flat array to minimize heap fragmentation
  // Layout: mel_filterbank_[filter_idx * N_FFT_BINS + bin_idx]
  this->mel_filterbank_.resize(N_MELS * N_FFT_BINS, 0.0f);

  float mel_min = this->hz_to_mel(0.0f);
  float mel_max = this->hz_to_mel(this->sample_rate_ / 2.0f);

  // Use stack arrays for temporary mel/bin points (small enough)
  float mel_points[N_MELS + 2];
  int bin_points[N_MELS + 2];

  for (int i = 0; i < N_MELS + 2; i++) {
    mel_points[i] = mel_min + i * (mel_max - mel_min) / (N_MELS + 1);
    float hz = this->mel_to_hz(mel_points[i]);
    bin_points[i] = (int)floorf((N_FFT + 1) * hz / this->sample_rate_);
  }

  // Create triangular filters in flat array
  for (int m = 0; m < N_MELS; m++) {
    int filter_offset = m * N_FFT_BINS;

    // Rising edge
    for (int k = bin_points[m]; k < bin_points[m + 1]; k++) {
      if (k < N_FFT_BINS && bin_points[m + 1] != bin_points[m]) {
        this->mel_filterbank_[filter_offset + k] =
            (float)(k - bin_points[m]) / (bin_points[m + 1] - bin_points[m]);
      }
    }
    // Falling edge
    for (int k = bin_points[m + 1]; k < bin_points[m + 2]; k++) {
      if (k < N_FFT_BINS && bin_points[m + 2] != bin_points[m + 1]) {
        this->mel_filterbank_[filter_offset + k] =
            (float)(bin_points[m + 2] - k) / (bin_points[m + 2] - bin_points[m + 1]);
      }
    }
  }

  ESP_LOGD(TAG, "Mel filterbank initialized: %d filters x %d bins = %d bytes",
           N_MELS, N_FFT_BINS, N_MELS * N_FFT_BINS * 4);
}

float BeepDetectorNNComponent::hz_to_mel(float hz) {
  return 2595.0f * log10f(1.0f + hz / 700.0f);
}

float BeepDetectorNNComponent::mel_to_hz(float mel) {
  return 700.0f * (powf(10.0f, mel / 2595.0f) - 1.0f);
}

float BeepDetectorNNComponent::run_inference(const float *mfcc_input) {
  // Manual neural network forward pass (v2 - simplified 2-layer model)
  // Architecture: Conv1D(8) -> BN -> ReLU -> Pool ->
  //               Conv1D(8) -> BN -> ReLU -> GAP ->
  //               Dense(8) -> ReLU -> Dense(1) -> Sigmoid

  float *buf1 = this->layer_buffer_1_.data();
  float *buf2 = this->layer_buffer_2_.data();

  // Layer 1: Conv1D(8) + BatchNorm + ReLU
  // Input: (25, 20), Output: (25, 8)
  this->conv1d_bn_relu(mfcc_input, buf1, MODEL_INPUT_FRAMES, 20, CONV1_FILTERS,
                       conv1d_kernel, conv1d_bias,
                       batch_normalization_gamma, batch_normalization_beta,
                       batch_normalization_mean, batch_normalization_var);

  // MaxPool1D: (25, 8) -> (12, 8)
  this->maxpool1d(buf1, buf2, MODEL_INPUT_FRAMES, CONV1_FILTERS);

  // Layer 2: Conv1D(8) + BatchNorm + ReLU
  // Input: (12, 8), Output: (12, 8)
  int pool1_len = MODEL_INPUT_FRAMES / 2;  // 12
  this->conv1d_bn_relu(buf2, buf1, pool1_len, CONV1_FILTERS, CONV2_FILTERS,
                       conv1d_1_kernel, conv1d_1_bias,
                       batch_normalization_1_gamma, batch_normalization_1_beta,
                       batch_normalization_1_mean, batch_normalization_1_var);

  // GlobalAveragePooling1D: (12, 8) -> (8,)
  this->global_avg_pool1d(buf1, buf2, pool1_len, CONV2_FILTERS);

  // Dense(8) + ReLU (skip dropout at inference)
  this->dense_relu(buf2, buf1, DENSE_UNITS, DENSE_UNITS, dense_kernel, dense_bias);

  // Dense(1) + Sigmoid
  float output;
  this->dense_sigmoid(buf1, &output, DENSE_UNITS, 1, dense_1_kernel, dense_1_bias);

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

    // Feed watchdog every 8 time steps
    if ((t & 0x07) == 0) {
      App.feed_wdt();
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
