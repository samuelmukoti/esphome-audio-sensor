// Auto generated code by esphome
// ========== AUTO GENERATED INCLUDE BLOCK BEGIN ===========
#include "esphome.h"
using namespace esphome;
using std::isnan;
using std::min;
using std::max;
using namespace microphone;
using namespace switch_;
using namespace button;
using namespace sensor;
using namespace binary_sensor;
logger::Logger *logger_logger_id;
wifi::WiFiComponent *wifi_wificomponent_id;
mdns::MDNSComponent *mdns_mdnscomponent_id;
esphome::ESPHomeOTAComponent *esphome_esphomeotacomponent_id;
safe_mode::SafeModeComponent *safe_mode_safemodecomponent_id;
api::APIServer *api_apiserver_id;
using namespace api;
i2s_audio::I2SAudioComponent *i2s_mic_bus;
i2s_audio::I2SAudioMicrophone *echo_microphone;
beep_detector_nn::BeepDetectorNNComponent *beep_detector_nn_component;
binary_sensor::BinarySensor *binary_sensor_binarysensor_id;
binary_sensor::DelayedOffFilter *binary_sensor_delayedofffilter_id;
binary_sensor::PressTrigger *binary_sensor_presstrigger_id;
Automation<> *automation_id;
api::HomeAssistantServiceCallAction<> *api_homeassistantservicecallaction_id;
sensor::Sensor *sensor_sensor_id;
sensor::Sensor *sensor_sensor_id_2;
template_::TemplateSwitch *nn_detection_enabled;
Automation<> *automation_id_3;
LambdaAction<> *lambdaaction_id_2;
Automation<> *automation_id_2;
LambdaAction<> *lambdaaction_id;
template_::TemplateButton *template__templatebutton_id;
button::ButtonPressTrigger *button_buttonpresstrigger_id;
Automation<> *automation_id_4;
LambdaAction<> *lambdaaction_id_3;
restart::RestartButton *restart_restartbutton_id;
wifi_signal::WiFiSignalSensor *wifi_signal_wifisignalsensor_id;
uptime::UptimeSecondsSensor *uptime_uptimesecondssensor_id;
preferences::IntervalSyncer *preferences_intervalsyncer_id;
// ========== AUTO GENERATED INCLUDE BLOCK END ==========="

void setup() {
  // ========== AUTO GENERATED CODE BEGIN ===========
  // network:
  //   enable_ipv6: false
  //   min_ipv6_addr_count: 0
  // esphome:
  //   name: esphome-web-d4d5d0
  //   friendly_name: m5Stack Echo d4d5d0
  //   min_version: 2025.9.0
  //   name_add_mac_suffix: false
  //   build_path: build/esphome-web-d4d5d0
  //   platformio_options: {}
  //   includes: []
  //   libraries: []
  //   debug_scheduler: false
  //   areas: []
  //   devices: []
  App.pre_setup("esphome-web-d4d5d0", "m5Stack Echo d4d5d0", "", __DATE__ ", " __TIME__, false);
  // microphone:
  // switch:
  // button:
  // sensor:
  // binary_sensor:
  // logger:
  //   level: DEBUG
  //   logs:
  //     beep_detector_nn: DEBUG
  //     i2s_audio: DEBUG
  //   id: logger_logger_id
  //   baud_rate: 115200
  //   tx_buffer_size: 512
  //   deassert_rts_dtr: false
  //   task_log_buffer_size: 768
  //   hardware_uart: UART0
  logger_logger_id = new logger::Logger(115200, 512);
  logger_logger_id->create_pthread_key();
  logger_logger_id->init_log_buffer(768);
  logger_logger_id->set_log_level(ESPHOME_LOG_LEVEL_DEBUG);
  logger_logger_id->set_uart_selection(logger::UART_SELECTION_UART0);
  logger_logger_id->pre_setup();
  logger_logger_id->set_log_level("beep_detector_nn", ESPHOME_LOG_LEVEL_DEBUG);
  logger_logger_id->set_log_level("i2s_audio", ESPHOME_LOG_LEVEL_DEBUG);
  logger_logger_id->set_component_source(LOG_STR("logger"));
  App.register_component(logger_logger_id);
  // wifi:
  //   id: wifi_wificomponent_id
  //   domain: .local
  //   reboot_timeout: 15min
  //   power_save_mode: LIGHT
  //   fast_connect: false
  //   enable_btm: false
  //   enable_rrm: false
  //   passive_scan: false
  //   enable_on_boot: true
  //   networks:
  //     - ssid: !secret 'wifi_ssid'
  //       password: !secret 'wifi_password'
  //       id: wifi_wifiap_id
  //       priority: 0.0
  //   use_address: esphome-web-d4d5d0.local
  wifi_wificomponent_id = new wifi::WiFiComponent();
  wifi_wificomponent_id->set_use_address("esphome-web-d4d5d0.local");
  {
  wifi::WiFiAP wifi_wifiap_id = wifi::WiFiAP();
  wifi_wifiap_id.set_ssid("YourWiFiSSID");
  wifi_wifiap_id.set_password("YourWiFiPassword");
  wifi_wifiap_id.set_priority(0.0f);
  wifi_wificomponent_id->add_sta(wifi_wifiap_id);
  }
  wifi_wificomponent_id->set_reboot_timeout(900000);
  wifi_wificomponent_id->set_power_save_mode(wifi::WIFI_POWER_SAVE_LIGHT);
  wifi_wificomponent_id->set_fast_connect(false);
  wifi_wificomponent_id->set_passive_scan(false);
  wifi_wificomponent_id->set_enable_on_boot(true);
  wifi_wificomponent_id->set_component_source(LOG_STR("wifi"));
  App.register_component(wifi_wificomponent_id);
  // mdns:
  //   id: mdns_mdnscomponent_id
  //   disabled: false
  //   services: []
  mdns_mdnscomponent_id = new mdns::MDNSComponent();
  mdns_mdnscomponent_id->set_component_source(LOG_STR("mdns"));
  App.register_component(mdns_mdnscomponent_id);
  // ota:
  // ota.esphome:
  //   platform: esphome
  //   id: esphome_esphomeotacomponent_id
  //   version: 2
  //   port: 3232
  esphome_esphomeotacomponent_id = new esphome::ESPHomeOTAComponent();
  esphome_esphomeotacomponent_id->set_port(3232);
  esphome_esphomeotacomponent_id->set_component_source(LOG_STR("esphome.ota"));
  App.register_component(esphome_esphomeotacomponent_id);
  // safe_mode:
  //   id: safe_mode_safemodecomponent_id
  //   boot_is_good_after: 1min
  //   disabled: false
  //   num_attempts: 10
  //   reboot_timeout: 5min
  safe_mode_safemodecomponent_id = new safe_mode::SafeModeComponent();
  safe_mode_safemodecomponent_id->set_component_source(LOG_STR("safe_mode"));
  App.register_component(safe_mode_safemodecomponent_id);
  if (safe_mode_safemodecomponent_id->should_enter_safe_mode(10, 300000, 60000)) return;
  // api:
  //   id: api_apiserver_id
  //   port: 6053
  //   password: ''
  //   reboot_timeout: 15min
  //   batch_delay: 100ms
  //   custom_services: false
  //   homeassistant_services: false
  //   homeassistant_states: false
  api_apiserver_id = new api::APIServer();
  api_apiserver_id->set_component_source(LOG_STR("api"));
  App.register_component(api_apiserver_id);
  api_apiserver_id->set_port(6053);
  api_apiserver_id->set_reboot_timeout(900000);
  api_apiserver_id->set_batch_delay(100);
  // esp32:
  //   variant: ESP32
  //   framework:
  //     version: 5.4.2
  //     sdkconfig_options: {}
  //     log_level: ERROR
  //     advanced:
  //       compiler_optimization: SIZE
  //       enable_lwip_assert: true
  //       ignore_efuse_custom_mac: false
  //       enable_lwip_mdns_queries: true
  //       enable_lwip_bridge_interface: false
  //       enable_lwip_tcpip_core_locking: true
  //       enable_lwip_check_thread_safety: true
  //     components: []
  //     platform_version: https:github.com/pioarduino/platform-espressif32/releases/download/54.03.21-2/platform-espressif32.zip
  //     source: pioarduino/framework-espidf@https:github.com/pioarduino/esp-idf/releases/download/v5.4.2/esp-idf-v5.4.2.zip
  //     type: esp-idf
  //   flash_size: 4MB
  //   board: esp32dev
  //   cpu_frequency: 160MHZ
  // external_components:
  //   - source:
  //       url: https:github.com/samuelmukoti/esphome-audio-sensor
  //       ref: main
  //       type: git
  //     components:
  //       - beep_detector_nn
  //     refresh: 0s
  // i2s_audio:
  //   id: i2s_mic_bus
  //   i2s_lrclk_pin: 33
  //   i2s_bclk_pin: 19
  i2s_mic_bus = new i2s_audio::I2SAudioComponent();
  i2s_mic_bus->set_component_source(LOG_STR("i2s_audio"));
  App.register_component(i2s_mic_bus);
  i2s_mic_bus->set_lrclk_pin(33);
  i2s_mic_bus->set_bclk_pin(19);
  // microphone.i2s_audio:
  //   platform: i2s_audio
  //   id: echo_microphone
  //   i2s_audio_id: i2s_mic_bus
  //   i2s_din_pin: 23
  //   pdm: true
  //   sample_rate: 16000
  //   bits_per_sample: 16.0
  //   channel: left
  //   i2s_mode: primary
  //   use_apll: false
  //   bits_per_channel: default
  //   mclk_multiple: 256
  //   correct_dc_offset: false
  //   adc_type: external
  //   num_channels: 1
  //   min_bits_per_sample: 16.0
  //   max_bits_per_sample: 16.0
  //   min_channels: 1
  //   max_channels: 1
  //   min_sample_rate: 16000
  //   max_sample_rate: 16000
  echo_microphone = new i2s_audio::I2SAudioMicrophone();
  echo_microphone->set_component_source(LOG_STR("i2s_audio.microphone"));
  App.register_component(echo_microphone);
  echo_microphone->set_parent(i2s_mic_bus);
  echo_microphone->set_i2s_role(::I2S_ROLE_MASTER);
  echo_microphone->set_slot_mode(::I2S_SLOT_MODE_MONO);
  echo_microphone->set_std_slot_mask(::I2S_STD_SLOT_LEFT);
  echo_microphone->set_slot_bit_width(::I2S_SLOT_BIT_WIDTH_16BIT);
  echo_microphone->set_sample_rate(16000);
  echo_microphone->set_use_apll(false);
  echo_microphone->set_mclk_multiple(::I2S_MCLK_MULTIPLE_256);
  echo_microphone->set_din_pin(23);
  echo_microphone->set_pdm(true);
  echo_microphone->set_correct_dc_offset(false);
  // beep_detector_nn:
  //   id: beep_detector_nn_component
  //   microphone: echo_microphone
  //   sample_rate: 16000
  //   window_size_ms: 250
  //   confidence_threshold: 0.7
  //   debounce_count: 2
  //   binary_sensor:
  //     name: Beep Detected (NN)
  //     device_class: sound
  //     filters:
  //       - delayed_off: 100ms
  //         type_id: binary_sensor_delayedofffilter_id
  //     on_press:
  //       - then:
  //           - homeassistant.event:
  //               event: esphome.beep_detected
  //               data:
  //                 device: m5stack_echo
  //                 message: Beep detected by neural network!
  //               id: api_apiserver_id
  //               data_template: {}
  //               variables: {}
  //             type_id: api_homeassistantservicecallaction_id
  //         automation_id: automation_id
  //         trigger_id: binary_sensor_presstrigger_id
  //     disabled_by_default: false
  //     id: binary_sensor_binarysensor_id
  //   confidence_sensor:
  //     name: Beep Confidence
  //     disabled_by_default: false
  //     id: sensor_sensor_id
  //     force_update: false
  //     unit_of_measurement: ''
  //     accuracy_decimals: 3
  //     state_class: measurement
  //   detection_count:
  //     name: Total Beep Detections (NN)
  //     disabled_by_default: false
  //     id: sensor_sensor_id_2
  //     force_update: false
  //     unit_of_measurement: ''
  //     accuracy_decimals: 0
  //     state_class: measurement
  beep_detector_nn_component = new beep_detector_nn::BeepDetectorNNComponent();
  beep_detector_nn_component->set_component_source(LOG_STR("beep_detector_nn"));
  App.register_component(beep_detector_nn_component);
  beep_detector_nn_component->set_microphone(echo_microphone);
  beep_detector_nn_component->set_sample_rate(16000);
  beep_detector_nn_component->set_confidence_threshold(0.7f);
  beep_detector_nn_component->set_window_size_ms(250);
  beep_detector_nn_component->set_debounce_count(2);
  binary_sensor_binarysensor_id = new binary_sensor::BinarySensor();
  App.register_binary_sensor(binary_sensor_binarysensor_id);
  binary_sensor_binarysensor_id->set_name("Beep Detected (NN)");
  binary_sensor_binarysensor_id->set_object_id("beep_detected__nn_");
  binary_sensor_binarysensor_id->set_disabled_by_default(false);
  binary_sensor_binarysensor_id->set_device_class("sound");
  binary_sensor_binarysensor_id->set_trigger_on_initial_state(false);
  binary_sensor_delayedofffilter_id = new binary_sensor::DelayedOffFilter();
  binary_sensor_delayedofffilter_id->set_component_source(LOG_STR("binary_sensor"));
  App.register_component(binary_sensor_delayedofffilter_id);
  binary_sensor_delayedofffilter_id->set_delay(100);
  binary_sensor_binarysensor_id->add_filters({binary_sensor_delayedofffilter_id});
  binary_sensor_presstrigger_id = new binary_sensor::PressTrigger(binary_sensor_binarysensor_id);
  automation_id = new Automation<>(binary_sensor_presstrigger_id);
  api_homeassistantservicecallaction_id = new api::HomeAssistantServiceCallAction<>(api_apiserver_id, true);
  api_homeassistantservicecallaction_id->set_service("esphome.beep_detected");
  api_homeassistantservicecallaction_id->add_data("device", "m5stack_echo");
  api_homeassistantservicecallaction_id->add_data("message", "Beep detected by neural network!");
  automation_id->add_actions({api_homeassistantservicecallaction_id});
  beep_detector_nn_component->set_binary_sensor(binary_sensor_binarysensor_id);
  sensor_sensor_id = new sensor::Sensor();
  App.register_sensor(sensor_sensor_id);
  sensor_sensor_id->set_name("Beep Confidence");
  sensor_sensor_id->set_object_id("beep_confidence");
  sensor_sensor_id->set_disabled_by_default(false);
  sensor_sensor_id->set_state_class(sensor::STATE_CLASS_MEASUREMENT);
  sensor_sensor_id->set_unit_of_measurement("");
  sensor_sensor_id->set_accuracy_decimals(3);
  sensor_sensor_id->set_force_update(false);
  beep_detector_nn_component->set_confidence_sensor(sensor_sensor_id);
  sensor_sensor_id_2 = new sensor::Sensor();
  App.register_sensor(sensor_sensor_id_2);
  sensor_sensor_id_2->set_name("Total Beep Detections (NN)");
  sensor_sensor_id_2->set_object_id("total_beep_detections__nn_");
  sensor_sensor_id_2->set_disabled_by_default(false);
  sensor_sensor_id_2->set_state_class(sensor::STATE_CLASS_MEASUREMENT);
  sensor_sensor_id_2->set_unit_of_measurement("");
  sensor_sensor_id_2->set_accuracy_decimals(0);
  sensor_sensor_id_2->set_force_update(false);
  beep_detector_nn_component->set_detection_count_sensor(sensor_sensor_id_2);
  // switch.template:
  //   platform: template
  //   name: NN Beep Detection Enabled
  //   id: nn_detection_enabled
  //   icon: mdi:brain
  //   optimistic: true
  //   restore_mode: RESTORE_DEFAULT_ON
  //   turn_on_action:
  //     then:
  //       - lambda: !lambda |-
  //           id(beep_detector_nn_component).set_enabled(true);
  //         type_id: lambdaaction_id
  //     trigger_id: trigger_id
  //     automation_id: automation_id_2
  //   turn_off_action:
  //     then:
  //       - lambda: !lambda |-
  //           id(beep_detector_nn_component).set_enabled(false);
  //         type_id: lambdaaction_id_2
  //     trigger_id: trigger_id_2
  //     automation_id: automation_id_3
  //   disabled_by_default: false
  //   assumed_state: false
  nn_detection_enabled = new template_::TemplateSwitch();
  App.register_switch(nn_detection_enabled);
  nn_detection_enabled->set_name("NN Beep Detection Enabled");
  nn_detection_enabled->set_object_id("nn_beep_detection_enabled");
  nn_detection_enabled->set_disabled_by_default(false);
  nn_detection_enabled->set_icon("mdi:brain");
  nn_detection_enabled->set_restore_mode(switch_::SWITCH_RESTORE_DEFAULT_ON);
  nn_detection_enabled->set_component_source(LOG_STR("template.switch"));
  App.register_component(nn_detection_enabled);
  automation_id_3 = new Automation<>(nn_detection_enabled->get_turn_off_trigger());
  lambdaaction_id_2 = new LambdaAction<>([=]() -> void {
      #line 105 "esphome-atom-d4d5d0.yaml"
      beep_detector_nn_component->set_enabled(false);
  });
  automation_id_3->add_actions({lambdaaction_id_2});
  automation_id_2 = new Automation<>(nn_detection_enabled->get_turn_on_trigger());
  lambdaaction_id = new LambdaAction<>([=]() -> void {
      #line 103 "esphome-atom-d4d5d0.yaml"
      beep_detector_nn_component->set_enabled(true);
  });
  automation_id_2->add_actions({lambdaaction_id});
  nn_detection_enabled->set_optimistic(true);
  nn_detection_enabled->set_assumed_state(false);
  // button.template:
  //   platform: template
  //   name: Reset NN Detection Count
  //   icon: mdi:counter
  //   on_press:
  //     - then:
  //         - lambda: !lambda |-
  //             id(beep_detector_nn_component).reset_detection_count();
  //           type_id: lambdaaction_id_3
  //       automation_id: automation_id_4
  //       trigger_id: button_buttonpresstrigger_id
  //   disabled_by_default: false
  //   id: template__templatebutton_id
  template__templatebutton_id = new template_::TemplateButton();
  App.register_button(template__templatebutton_id);
  template__templatebutton_id->set_name("Reset NN Detection Count");
  template__templatebutton_id->set_object_id("reset_nn_detection_count");
  template__templatebutton_id->set_disabled_by_default(false);
  template__templatebutton_id->set_icon("mdi:counter");
  button_buttonpresstrigger_id = new button::ButtonPressTrigger(template__templatebutton_id);
  automation_id_4 = new Automation<>(button_buttonpresstrigger_id);
  lambdaaction_id_3 = new LambdaAction<>([=]() -> void {
      #line 112 "esphome-atom-d4d5d0.yaml"
      beep_detector_nn_component->reset_detection_count();
  });
  automation_id_4->add_actions({lambdaaction_id_3});
  // button.restart:
  //   platform: restart
  //   name: Restart Device
  //   disabled_by_default: false
  //   id: restart_restartbutton_id
  //   icon: mdi:restart
  //   entity_category: config
  //   device_class: restart
  restart_restartbutton_id = new restart::RestartButton();
  restart_restartbutton_id->set_component_source(LOG_STR("restart.button"));
  App.register_component(restart_restartbutton_id);
  App.register_button(restart_restartbutton_id);
  restart_restartbutton_id->set_name("Restart Device");
  restart_restartbutton_id->set_object_id("restart_device");
  restart_restartbutton_id->set_disabled_by_default(false);
  restart_restartbutton_id->set_icon("mdi:restart");
  restart_restartbutton_id->set_entity_category(::ENTITY_CATEGORY_CONFIG);
  restart_restartbutton_id->set_device_class("restart");
  // sensor.wifi_signal:
  //   platform: wifi_signal
  //   name: WiFi Signal
  //   update_interval: 60s
  //   disabled_by_default: false
  //   force_update: false
  //   id: wifi_signal_wifisignalsensor_id
  //   unit_of_measurement: dBm
  //   accuracy_decimals: 0
  //   device_class: signal_strength
  //   state_class: measurement
  //   entity_category: diagnostic
  wifi_signal_wifisignalsensor_id = new wifi_signal::WiFiSignalSensor();
  App.register_sensor(wifi_signal_wifisignalsensor_id);
  wifi_signal_wifisignalsensor_id->set_name("WiFi Signal");
  wifi_signal_wifisignalsensor_id->set_object_id("wifi_signal");
  wifi_signal_wifisignalsensor_id->set_disabled_by_default(false);
  wifi_signal_wifisignalsensor_id->set_entity_category(::ENTITY_CATEGORY_DIAGNOSTIC);
  wifi_signal_wifisignalsensor_id->set_device_class("signal_strength");
  wifi_signal_wifisignalsensor_id->set_state_class(sensor::STATE_CLASS_MEASUREMENT);
  wifi_signal_wifisignalsensor_id->set_unit_of_measurement("dBm");
  wifi_signal_wifisignalsensor_id->set_accuracy_decimals(0);
  wifi_signal_wifisignalsensor_id->set_force_update(false);
  wifi_signal_wifisignalsensor_id->set_update_interval(60000);
  wifi_signal_wifisignalsensor_id->set_component_source(LOG_STR("wifi_signal.sensor"));
  App.register_component(wifi_signal_wifisignalsensor_id);
  // sensor.uptime:
  //   platform: uptime
  //   name: Uptime
  //   update_interval: 60s
  //   disabled_by_default: false
  //   force_update: false
  //   id: uptime_uptimesecondssensor_id
  //   unit_of_measurement: s
  //   icon: mdi:timer-outline
  //   accuracy_decimals: 0
  //   device_class: duration
  //   state_class: total_increasing
  //   entity_category: diagnostic
  //   type: seconds
  uptime_uptimesecondssensor_id = new uptime::UptimeSecondsSensor();
  App.register_sensor(uptime_uptimesecondssensor_id);
  uptime_uptimesecondssensor_id->set_name("Uptime");
  uptime_uptimesecondssensor_id->set_object_id("uptime");
  uptime_uptimesecondssensor_id->set_disabled_by_default(false);
  uptime_uptimesecondssensor_id->set_icon("mdi:timer-outline");
  uptime_uptimesecondssensor_id->set_entity_category(::ENTITY_CATEGORY_DIAGNOSTIC);
  uptime_uptimesecondssensor_id->set_device_class("duration");
  uptime_uptimesecondssensor_id->set_state_class(sensor::STATE_CLASS_TOTAL_INCREASING);
  uptime_uptimesecondssensor_id->set_unit_of_measurement("s");
  uptime_uptimesecondssensor_id->set_accuracy_decimals(0);
  uptime_uptimesecondssensor_id->set_force_update(false);
  uptime_uptimesecondssensor_id->set_update_interval(60000);
  uptime_uptimesecondssensor_id->set_component_source(LOG_STR("uptime.sensor"));
  App.register_component(uptime_uptimesecondssensor_id);
  // preferences:
  //   id: preferences_intervalsyncer_id
  //   flash_write_interval: 60s
  preferences_intervalsyncer_id = new preferences::IntervalSyncer();
  preferences_intervalsyncer_id->set_write_interval(60000);
  preferences_intervalsyncer_id->set_component_source(LOG_STR("preferences"));
  App.register_component(preferences_intervalsyncer_id);
  // socket:
  //   implementation: bsd_sockets
  // md5:
  // audio:
  //   {}
  // =========== AUTO GENERATED CODE END ============
  App.setup();
}

void loop() {
  App.loop();
}
