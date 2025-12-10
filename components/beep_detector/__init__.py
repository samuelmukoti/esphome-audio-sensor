import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, sensor, microphone
from esphome.const import (
    CONF_ID,
    CONF_MICROPHONE,
    STATE_CLASS_MEASUREMENT,
)

DEPENDENCIES = ["i2s_audio"]
AUTO_LOAD = ["binary_sensor", "sensor"]

beep_detector_ns = cg.esphome_ns.namespace("beep_detector")
BeepDetectorComponent = beep_detector_ns.class_("BeepDetectorComponent", cg.Component)

# Configuration keys
CONF_TARGET_FREQUENCY = "target_frequency"
CONF_SAMPLE_RATE = "sample_rate"
CONF_WINDOW_SIZE = "window_size"
CONF_ENERGY_THRESHOLD = "energy_threshold"
CONF_RMS_THRESHOLD = "rms_threshold"
CONF_MIN_DURATION = "min_duration"
CONF_MAX_DURATION = "max_duration"
CONF_COOLDOWN = "cooldown"
CONF_DEBOUNCE_COUNT = "debounce_count"
CONF_BINARY_SENSOR = "binary_sensor"
CONF_ENERGY_SENSOR = "energy_sensor"
CONF_RMS_SENSOR = "rms_sensor"
CONF_DETECTION_COUNT = "detection_count"

CONFIG_SCHEMA = cv.Schema({
    cv.GenerateID(): cv.declare_id(BeepDetectorComponent),
    cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),

    # Detection parameters
    cv.Optional(CONF_TARGET_FREQUENCY, default=2615.0): cv.float_range(min=100.0, max=8000.0),
    cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.int_range(min=8000, max=48000),
    cv.Optional(CONF_WINDOW_SIZE, default=100): cv.int_range(min=10, max=500),
    cv.Optional(CONF_ENERGY_THRESHOLD, default=100.0): cv.float_range(min=0.0),
    cv.Optional(CONF_RMS_THRESHOLD, default=0.0069): cv.float_range(min=0.0, max=1.0),
    cv.Optional(CONF_MIN_DURATION, default=40): cv.int_range(min=10, max=1000),
    cv.Optional(CONF_MAX_DURATION, default=100): cv.int_range(min=10, max=1000),
    cv.Optional(CONF_COOLDOWN, default=200): cv.int_range(min=0, max=5000),
    cv.Optional(CONF_DEBOUNCE_COUNT, default=2): cv.int_range(min=1, max=10),

    # Sensors
    cv.Optional(CONF_BINARY_SENSOR): binary_sensor.binary_sensor_schema(),
    cv.Optional(CONF_ENERGY_SENSOR): sensor.sensor_schema(
        unit_of_measurement="",
        accuracy_decimals=2,
        state_class=STATE_CLASS_MEASUREMENT,
        icon="mdi:volume-high",
    ),
    cv.Optional(CONF_RMS_SENSOR): sensor.sensor_schema(
        unit_of_measurement="",
        accuracy_decimals=4,
        state_class=STATE_CLASS_MEASUREMENT,
        icon="mdi:volume-high",
    ),
    cv.Optional(CONF_DETECTION_COUNT): sensor.sensor_schema(
        unit_of_measurement="",
        accuracy_decimals=0,
        state_class=STATE_CLASS_MEASUREMENT,
        icon="mdi:counter",
    ),
}).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    # Set microphone
    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    # Set detection parameters
    cg.add(var.set_target_frequency(config[CONF_TARGET_FREQUENCY]))
    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_window_size_ms(config[CONF_WINDOW_SIZE]))
    cg.add(var.set_energy_threshold(config[CONF_ENERGY_THRESHOLD]))
    cg.add(var.set_rms_threshold(config[CONF_RMS_THRESHOLD]))
    cg.add(var.set_min_duration_ms(config[CONF_MIN_DURATION]))
    cg.add(var.set_max_duration_ms(config[CONF_MAX_DURATION]))
    cg.add(var.set_cooldown_ms(config[CONF_COOLDOWN]))
    cg.add(var.set_debounce_count(config[CONF_DEBOUNCE_COUNT]))

    # Register binary sensor
    if CONF_BINARY_SENSOR in config:
        sens = await binary_sensor.new_binary_sensor(config[CONF_BINARY_SENSOR])
        cg.add(var.set_binary_sensor(sens))

    # Register energy sensor
    if CONF_ENERGY_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_ENERGY_SENSOR])
        cg.add(var.set_energy_sensor(sens))

    # Register RMS sensor
    if CONF_RMS_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_RMS_SENSOR])
        cg.add(var.set_rms_sensor(sens))

    # Register detection count sensor
    if CONF_DETECTION_COUNT in config:
        sens = await sensor.new_sensor(config[CONF_DETECTION_COUNT])
        cg.add(var.set_detection_count_sensor(sens))
