import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import binary_sensor, sensor
from esphome.const import (
    CONF_ID,
    DEVICE_CLASS_SOUND,
    STATE_CLASS_MEASUREMENT,
    UNIT_EMPTY,
)

DEPENDENCIES = []
AUTO_LOAD = ["binary_sensor", "sensor"]

CONF_PORT = "port"
CONF_BINARY_SENSOR = "binary_sensor"
CONF_CONFIDENCE_SENSOR = "confidence_sensor"
CONF_DETECTION_COUNT = "detection_count"

detection_receiver_ns = cg.esphome_ns.namespace("detection_receiver")
DetectionReceiverComponent = detection_receiver_ns.class_(
    "DetectionReceiverComponent", cg.Component
)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(DetectionReceiverComponent),
        cv.Optional(CONF_PORT, default=5001): cv.port,
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

    cg.add(var.set_port(config[CONF_PORT]))

    if CONF_BINARY_SENSOR in config:
        sens = await binary_sensor.new_binary_sensor(config[CONF_BINARY_SENSOR])
        cg.add(var.set_binary_sensor(sens))

    if CONF_CONFIDENCE_SENSOR in config:
        sens = await sensor.new_sensor(config[CONF_CONFIDENCE_SENSOR])
        cg.add(var.set_confidence_sensor(sens))

    if CONF_DETECTION_COUNT in config:
        sens = await sensor.new_sensor(config[CONF_DETECTION_COUNT])
        cg.add(var.set_detection_count_sensor(sens))
