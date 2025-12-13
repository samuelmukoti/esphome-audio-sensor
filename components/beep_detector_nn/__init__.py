import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import microphone, binary_sensor, sensor
from esphome.const import (
    CONF_ID,
    CONF_MICROPHONE,
    DEVICE_CLASS_SOUND,
    STATE_CLASS_MEASUREMENT,
    UNIT_EMPTY,
)

DEPENDENCIES = ["microphone"]
AUTO_LOAD = ["binary_sensor", "sensor"]

CONF_BINARY_SENSOR = "binary_sensor"
CONF_CONFIDENCE_SENSOR = "confidence_sensor"
CONF_DETECTION_COUNT = "detection_count"
CONF_SAMPLE_RATE = "sample_rate"
CONF_CONFIDENCE_THRESHOLD = "confidence_threshold"
CONF_WINDOW_SIZE_MS = "window_size_ms"
CONF_DEBOUNCE_COUNT = "debounce_count"

beep_detector_nn_ns = cg.esphome_ns.namespace("beep_detector_nn")
BeepDetectorNNComponent = beep_detector_nn_ns.class_(
    "BeepDetectorNNComponent", cg.Component
)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(BeepDetectorNNComponent),
        cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.positive_int,
        cv.Optional(CONF_CONFIDENCE_THRESHOLD, default=0.7): cv.float_range(min=0.0, max=1.0),
        cv.Optional(CONF_WINDOW_SIZE_MS, default=500): cv.positive_int,
        cv.Optional(CONF_DEBOUNCE_COUNT, default=2): cv.positive_int,
        cv.Optional(CONF_BINARY_SENSOR): binary_sensor.binary_sensor_schema(
            device_class=DEVICE_CLASS_SOUND
        ),
        cv.Optional(CONF_CONFIDENCE_SENSOR): sensor.sensor_schema(
            unit_of_measurement=UNIT_EMPTY,
            accuracy_decimals=3,
            state_class=STATE_CLASS_MEASUREMENT,
        ),
        cv.Optional(CONF_DETECTION_COUNT): sensor.sensor_schema(
            unit_of_measurement=UNIT_EMPTY,
            accuracy_decimals=0,
            state_class=STATE_CLASS_MEASUREMENT,
        ),
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_confidence_threshold(config[CONF_CONFIDENCE_THRESHOLD]))
    cg.add(var.set_window_size_ms(config[CONF_WINDOW_SIZE_MS]))
    cg.add(var.set_debounce_count(config[CONF_DEBOUNCE_COUNT]))

    if CONF_BINARY_SENSOR in config:
        sens = await binary_sensor.new_binary_sensor(config[CONF_BINARY_SENSOR])
        cg.add(var.set_binary_sensor(sens))

    if CONF_CONFIDENCE_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_CONFIDENCE_SENSOR])
        cg.add(var.set_confidence_sensor(sens))

    if CONF_DETECTION_COUNT in config:
        sens = await sensor.new_sensor(config[CONF_DETECTION_COUNT])
        cg.add(var.set_detection_count_sensor(sens))

    # No external dependencies needed - using manual NN inference
