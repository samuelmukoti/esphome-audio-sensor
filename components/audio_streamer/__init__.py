import esphome.codegen as cg
import esphome.config_validation as cv
from esphome.components import microphone
from esphome.const import CONF_ID, CONF_MICROPHONE

DEPENDENCIES = ["microphone", "wifi"]
AUTO_LOAD = ["socket"]

CONF_TARGET_IP = "target_ip"
CONF_TARGET_PORT = "target_port"
CONF_SAMPLE_RATE = "sample_rate"
CONF_ENABLED = "enabled"
CONF_CHUNK_SIZE = "chunk_size"
CONF_BATCH_COUNT = "batch_count"
CONF_SEND_INTERVAL_MS = "send_interval_ms"

audio_streamer_ns = cg.esphome_ns.namespace("audio_streamer")
AudioStreamerComponent = audio_streamer_ns.class_(
    "AudioStreamerComponent", cg.Component
)

CONFIG_SCHEMA = cv.Schema(
    {
        cv.GenerateID(): cv.declare_id(AudioStreamerComponent),
        cv.Required(CONF_MICROPHONE): cv.use_id(microphone.Microphone),
        cv.Required(CONF_TARGET_IP): cv.string,
        cv.Optional(CONF_TARGET_PORT, default=5000): cv.port,
        cv.Optional(CONF_SAMPLE_RATE, default=16000): cv.positive_int,
        cv.Optional(CONF_ENABLED, default=False): cv.boolean,
        cv.Optional(CONF_CHUNK_SIZE, default=512): cv.positive_int,
        # Flow control options to prevent WiFi buffer exhaustion
        cv.Optional(CONF_BATCH_COUNT, default=4): cv.int_range(min=1, max=16),  # Batch N reads before sending
        cv.Optional(CONF_SEND_INTERVAL_MS, default=50): cv.int_range(min=10, max=500),  # Min ms between sends
    }
).extend(cv.COMPONENT_SCHEMA)


async def to_code(config):
    var = cg.new_Pvariable(config[CONF_ID])
    await cg.register_component(var, config)

    mic = await cg.get_variable(config[CONF_MICROPHONE])
    cg.add(var.set_microphone(mic))

    cg.add(var.set_target_ip(config[CONF_TARGET_IP]))
    cg.add(var.set_target_port(config[CONF_TARGET_PORT]))
    cg.add(var.set_sample_rate(config[CONF_SAMPLE_RATE]))
    cg.add(var.set_enabled(config[CONF_ENABLED]))
    cg.add(var.set_chunk_size(config[CONF_CHUNK_SIZE]))
    cg.add(var.set_batch_count(config[CONF_BATCH_COUNT]))
    cg.add(var.set_send_interval_ms(config[CONF_SEND_INTERVAL_MS]))
