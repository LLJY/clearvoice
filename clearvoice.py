#!/usr/bin/env python3
"""ClearVoice — PipeWire noise cancellation, beamforming & AEC system tray tool.

Creates a virtual microphone with DeepFilterNet noise cancellation,
WebRTC-based beamforming, and acoustic echo cancellation via PipeWire.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib

try:
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator

    HAS_APPINDICATOR = True
except (ValueError, ImportError):
    try:
        gi.require_version("AppIndicator3", "0.1")
        from gi.repository import AppIndicator3 as AppIndicator

        HAS_APPINDICATOR = True
    except (ValueError, ImportError):
        HAS_APPINDICATOR = False


# ── Constants ─────────────────────────────────────────────────────────────────

APP_ID = "clearvoice"
APP_NAME = "ClearVoice"
VERSION = "0.1.0"

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / APP_ID
CONFIG_FILE = CONFIG_DIR / "config.json"
LOG_FILE = CONFIG_DIR / "clearvoice.log"

RUNTIME_DIR = (
    Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/run-{os.getuid()}")) / APP_ID
)

# PipeWire node names
VIRTUAL_MIC_NAME = "clearvoice_source"
VIRTUAL_MIC_DESC = "ClearVoice"
EC_SOURCE_NAME = "clearvoice_beamformed"
EC_SOURCE_DESC = "ClearVoice Beamformed"

# LADSPA plugin
DEEPFILTER_SO = "libdeep_filter_ladspa.so"
DEEPFILTER_LABEL_MONO = "deep_filter_mono"
DEEPFILTER_LABEL_STEREO = "deep_filter_stereo"

LADSPA_SEARCH_PATHS = [
    "/usr/lib/ladspa",
    "/usr/lib64/ladspa",
    "/usr/local/lib/ladspa",
    str(Path.home() / ".ladspa"),
]

# Speaker enhancement config (ships with the project)
SPEAKER_CHAIN_CONF = Path(__file__).parent / "speaker-chain.conf"
SPEAKER_SINK_NAME = "clearvoice_speakers"

# Icons (3 states)
ICON_ACTIVE = "audio-input-microphone"  # black — processing audio
ICON_STANDBY = "audio-input-microphone-symbolic"  # grey — enabled, idle
ICON_OFF = "microphone-sensitivity-muted-symbolic"  # slashed — disabled

log = logging.getLogger(APP_ID)


# ── Mic Geometry Presets ──────────────────────────────────────────────────────

MIC_PRESETS = {
    "laptop-dual-60mm": {
        "label": "Dual 60mm (ThinkPad/Dell)",
        "geometry": "-0.03,0,0,0.03,0,0",
    },
    "laptop-dual-40mm": {
        "label": "Dual 40mm (Compact)",
        "geometry": "-0.02,0,0,0.02,0,0",
    },
    "laptop-triple-linear": {
        "label": "Triple Linear 40mm",
        "geometry": "-0.04,0,0,0,0,0,0.04,0,0",
    },
    "webcam-stereo": {
        "label": "Webcam Stereo 100mm",
        "geometry": "-0.05,0,0,0.05,0,0",
    },
}


# ── Default Config ────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "enabled": True,
    "source_device": None,
    "noise_cancellation": {
        "enabled": True,
        "attenuation_limit_db": 100,
        "min_processing_threshold_db": -15,
        "max_erb_threshold_db": 35,
        "max_df_threshold_db": 35,
        "post_filter_beta": 0.0,
    },
    "beamforming": {
        "enabled": False,
        "preset": "laptop-dual-60mm",
        "custom_geometry": None,
    },
    "echo_cancellation": {
        "enabled": False,
    },
    "speaker_enhancement": {
        "enabled": True,
    },
    "previous_default_source": None,
    "previous_default_sink": None,
}


# ── Config I/O ────────────────────────────────────────────────────────────────


def load_config() -> dict:
    """Load config from disk, merged with defaults."""
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            _deep_merge(config, saved)
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to load config: %s", exc)
    return config


def save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(config, f, indent=2)
    tmp.replace(CONFIG_FILE)


def _deep_merge(base: dict, override: dict):
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


# ── Dependency Detection ──────────────────────────────────────────────────────


def find_ladspa_plugin(filename: str) -> str | None:
    """Search standard paths + $LADSPA_PATH for a plugin .so."""
    ladspa_env = os.environ.get("LADSPA_PATH", "")
    search = ladspa_env.split(":") if ladspa_env else []
    search.extend(LADSPA_SEARCH_PATHS)
    for d in search:
        p = Path(d) / filename
        if p.is_file():
            return str(p)
    return None


def check_dependencies() -> list[str]:
    """Return list of missing dependencies."""
    missing = []
    for cmd in ("pipewire", "pw-dump", "pactl", "wpctl"):
        if not shutil.which(cmd):
            missing.append(cmd)
    if not find_ladspa_plugin(DEEPFILTER_SO):
        missing.append(f"DeepFilterNet LADSPA ({DEEPFILTER_SO})")
    return missing


# ── PipeWire Helpers ──────────────────────────────────────────────────────────


def pw_list_sources() -> list[dict]:
    """Enumerate physical audio sources via pw-dump."""
    try:
        result = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return []
        objects = json.loads(result.stdout)
        sources = []
        for obj in objects:
            if obj.get("type") != "PipeWire:Interface:Node":
                continue
            props = obj.get("info", {}).get("props", {})
            if props.get("media.class") != "Audio/Source":
                continue
            name = props.get("node.name", "")
            if name.startswith("clearvoice") or ".monitor" in name:
                continue
            sources.append(
                {
                    "id": obj["id"],
                    "name": name,
                    "description": props.get("node.description", name),
                }
            )
        return sources
    except Exception as exc:
        log.error("Failed to enumerate sources: %s", exc)
        return []


def pw_get_default_source() -> str:
    try:
        r = subprocess.run(
            ["pactl", "get-default-source"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def pw_get_default_sink() -> str:
    try:
        r = subprocess.run(
            ["pactl", "get-default-sink"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.stdout.strip()
    except Exception:
        return ""


def pw_set_default_sink(node_id: int) -> bool:
    """Set default sink by PipeWire node ID via wpctl."""
    try:
        r = subprocess.run(
            ["wpctl", "set-default", str(node_id)],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.returncode == 0
    except Exception:
        return False


def pw_find_node_id(node_name: str) -> int | None:
    """Find a PipeWire node ID by node.name."""
    try:
        r = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return None
        for obj in json.loads(r.stdout):
            props = obj.get("info", {}).get("props", {})
            if props.get("node.name") == node_name:
                if props.get("media.class") in ("Audio/Sink", "Audio/Source"):
                    return obj["id"]
        return None
    except Exception:
        return None


def pw_nodes_active(prefix: str = "clearvoice") -> bool:
    """Check if any clearvoice PipeWire nodes are in 'running' state."""
    try:
        r = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return False
        for obj in json.loads(r.stdout):
            props = obj.get("info", {}).get("props", {})
            name = props.get("node.name", "")
            state = obj.get("info", {}).get("state", "")
            if name.startswith(prefix) and state == "running":
                return True
        return False
    except Exception:
        return False


def pw_set_default_source(name: str) -> bool:
    try:
        r = subprocess.run(
            ["pactl", "set-default-source", name],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return r.returncode == 0
    except Exception:
        return False


def pw_node_exists(node_name: str) -> bool:
    try:
        r = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return False
        for obj in json.loads(r.stdout):
            props = obj.get("info", {}).get("props", {})
            if props.get("node.name") == node_name:
                return True
        return False
    except Exception:
        return False


def pw_wait_for_node(node_name: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pw_node_exists(node_name):
            return True
        time.sleep(0.25)
    return False


# ── PipeWire Config Generation ───────────────────────────────────────────────


def _pw_conf_filter_chain(
    plugin_path: str,
    attenuation_db: int = 100,
    min_proc_db: int = -15,
    max_erb_db: int = 35,
    max_df_db: int = 35,
    post_filter_beta: float = 0.0,
    target_source: str | None = None,
) -> str:
    """Build a PipeWire config that loads a DeepFilterNet filter-chain."""
    target_line = ""
    if target_source:
        target_line = f'target.object = "{target_source}"'

    return (
        "# ClearVoice filter-chain (auto-generated)\n"
        "context.properties = {\n"
        "    log.level = 0\n"
        "}\n"
        "\n"
        "context.spa-libs = {\n"
        "    audio.convert.* = audioconvert/libspa-audioconvert\n"
        "    support.*       = support/libspa-support\n"
        "}\n"
        "\n"
        "context.modules = [\n"
        "    { name = libpipewire-module-rt\n"
        "        args = { nice.level = -11 }\n"
        "        flags = [ ifexists nofail ]\n"
        "    }\n"
        "    { name = libpipewire-module-protocol-native }\n"
        "    { name = libpipewire-module-client-node }\n"
        "    { name = libpipewire-module-adapter }\n"
        "    { name = libpipewire-module-filter-chain\n"
        "        args = {\n"
        f'            node.description = "{VIRTUAL_MIC_DESC}"\n'
        f'            media.name       = "{VIRTUAL_MIC_DESC}"\n'
        "            filter.graph = {\n"
        "                nodes = [\n"
        "                    {\n"
        "                        type   = ladspa\n"
        "                        name   = deepfilter\n"
        f"                        plugin = {plugin_path}\n"
        f"                        label  = {DEEPFILTER_LABEL_MONO}\n"
        "                        control = {\n"
        f'                            "Attenuation Limit (dB)" = {attenuation_db}\n'
        f'                            "Min processing threshold (dB)" = {min_proc_db}\n'
        f'                            "Max ERB processing threshold (dB)" = {max_erb_db}\n'
        f'                            "Max DF processing threshold (dB)" = {max_df_db}\n'
        f'                            "Post Filter Beta" = {post_filter_beta}\n'
        "                        }\n"
        "                    }\n"
        "                ]\n"
        "            }\n"
        "            capture.props = {\n"
        f'                node.name    = "clearvoice_capture"\n'
        "                node.passive = true\n"
        "                audio.rate   = 48000\n"
        f"                {target_line}\n"
        "            }\n"
        "            playback.props = {\n"
        f'                node.name        = "{VIRTUAL_MIC_NAME}"\n'
        f'                node.description = "{VIRTUAL_MIC_DESC}"\n'
        "                media.class      = Audio/Source\n"
        "                audio.rate       = 48000\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "]\n"
    )


def _pw_conf_echo_cancel(
    target_source: str | None = None,
    beamforming: bool = False,
    mic_geometry: str = "",
    source_name: str = EC_SOURCE_NAME,
    source_desc: str = EC_SOURCE_DESC,
    is_intermediate: bool = False,
) -> str:
    """Build a PipeWire config that loads the echo-cancel module."""
    target_line = ""
    if target_source:
        target_line = f'target.object = "{target_source}"'

    aec_parts = []
    if beamforming and mic_geometry:
        aec_parts.append("beamforming=1")
        aec_parts.append(f"mic_geometry={mic_geometry}")
    else:
        aec_parts.append("beamforming=0")
    aec_args = " ".join(aec_parts)

    return (
        "# ClearVoice echo-cancel (auto-generated)\n"
        "context.properties = {\n"
        "    log.level = 0\n"
        "}\n"
        "\n"
        "context.spa-libs = {\n"
        "    audio.convert.* = audioconvert/libspa-audioconvert\n"
        "    support.*       = support/libspa-support\n"
        "    aec.*           = aec/libspa-aec-webrtc\n"
        "}\n"
        "\n"
        "context.modules = [\n"
        "    { name = libpipewire-module-rt\n"
        "        args = { nice.level = -11 }\n"
        "        flags = [ ifexists nofail ]\n"
        "    }\n"
        "    { name = libpipewire-module-protocol-native }\n"
        "    { name = libpipewire-module-client-node }\n"
        "    { name = libpipewire-module-adapter }\n"
        "    { name = libpipewire-module-echo-cancel\n"
        "        args = {\n"
        "            library.name = aec/libspa-aec-webrtc\n"
        f'            aec.args     = "{aec_args}"\n'
        "            capture.props = {\n"
        '                node.name = "clearvoice_ec_capture"\n'
        f"                {target_line}\n"
        "            }\n"
        "            source.props = {\n"
        f'                node.name        = "{source_name}"\n'
        f'                node.description = "{source_desc}"\n'
        "                media.class      = Audio/Source\n"
        f"                {'priority.session = 0' if is_intermediate else ''}\n"
        "            }\n"
        "            sink.props = {\n"
        '                node.name = "clearvoice_ec_sink"\n'
        "            }\n"
        "            playback.props = {\n"
        '                node.name    = "clearvoice_ec_playback"\n'
        "                node.passive = true\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "]\n"
    )


# ── Pipeline Manager ─────────────────────────────────────────────────────────


class PipelineManager:
    """Manages the ClearVoice audio processing pipeline.

    Spawns up to three PipeWire client processes:
      1. echo-cancel   (beamforming / AEC)   → optional intermediate source
      2. filter-chain   (DeepFilterNet)       → virtual mic
      3. speaker-chain  (EQ / bass / stereo)  → virtual sink for speakers
    """

    def __init__(self, config: dict):
        self.config = config
        self._fc_proc: subprocess.Popen | None = None
        self._ec_proc: subprocess.Popen | None = None
        self._spk_proc: subprocess.Popen | None = None
        self._running = False
        self._lock = threading.Lock()

    # ── Properties ──

    @property
    def running(self) -> bool:
        return self._running

    @property
    def nc_enabled(self) -> bool:
        return self.config["noise_cancellation"]["enabled"]

    @property
    def bf_enabled(self) -> bool:
        return self.config["beamforming"]["enabled"]

    @property
    def aec_enabled(self) -> bool:
        return self.config["echo_cancellation"]["enabled"]

    @property
    def ec_needed(self) -> bool:
        return self.bf_enabled or self.aec_enabled

    @property
    def spk_enabled(self) -> bool:
        return self.config.get("speaker_enhancement", {}).get("enabled", False)

    @property
    def any_processing(self) -> bool:
        return self.nc_enabled or self.ec_needed or self.spk_enabled

    # ── Source Resolution ──

    def _resolve_source(self) -> str | None:
        """Figure out which physical source to capture from."""
        available = {s["name"] for s in pw_list_sources()}

        selected = self.config.get("source_device")
        if selected:
            if selected in available:
                return selected
            log.warning("Configured source %s not found, falling back", selected)

        # Auto: use current default unless it's our own node
        current = pw_get_default_source()
        if current and not current.startswith("clearvoice"):
            if not available or current in available:
                return current

        # Fall back to stored previous
        prev = self.config.get("previous_default_source")
        if prev and prev in available:
            return prev

        # Last resort: first physical source
        return next(iter(available), None)

    # ── Start / Stop ──

    def start(self) -> tuple[bool, str]:
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> tuple[bool, str]:
        if self._running:
            return True, "Already running"

        if not self.any_processing:
            return False, "Enable at least one processing feature"

        needs_mic = self.nc_enabled or self.ec_needed
        source = None

        if needs_mic:
            plugin_path = find_ladspa_plugin(DEEPFILTER_SO)
            if self.nc_enabled and not plugin_path:
                return False, f"LADSPA plugin not found: {DEEPFILTER_SO}"

            source = self._resolve_source()
            if not source:
                return False, "No audio source device found"

        log.info(
            "Starting pipeline — source=%s mic=%s spk=%s",
            source,
            needs_mic,
            self.spk_enabled,
        )

        # Remember current defaults so we can restore them
        if needs_mic:
            current_default = pw_get_default_source()
            if current_default and not current_default.startswith("clearvoice"):
                self.config["previous_default_source"] = current_default
                save_config(self.config)

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        # Which node becomes the system default virtual mic?
        final_node = VIRTUAL_MIC_NAME if self.nc_enabled else VIRTUAL_MIC_NAME

        try:
            # ── Stage 1: Echo-cancel (beamforming / AEC) ──
            if self.ec_needed:
                if self.nc_enabled:
                    ec_out_name = EC_SOURCE_NAME
                    ec_out_desc = EC_SOURCE_DESC
                else:
                    # Echo-cancel IS the final stage
                    ec_out_name = VIRTUAL_MIC_NAME
                    ec_out_desc = VIRTUAL_MIC_DESC

                geometry = ""
                if self.bf_enabled:
                    custom = self.config["beamforming"].get("custom_geometry")
                    if custom:
                        geometry = custom
                    else:
                        preset = self.config["beamforming"].get(
                            "preset", "laptop-dual-60mm"
                        )
                        geometry = MIC_PRESETS.get(preset, {}).get("geometry", "")

                conf = _pw_conf_echo_cancel(
                    target_source=source,
                    beamforming=self.bf_enabled,
                    mic_geometry=geometry,
                    source_name=ec_out_name,
                    source_desc=ec_out_desc,
                    is_intermediate=self.nc_enabled,
                )
                conf_path = RUNTIME_DIR / "echo-cancel.conf"
                conf_path.write_text(conf)

                log.info("Spawning echo-cancel process")
                self._ec_proc = subprocess.Popen(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                if not pw_wait_for_node(ec_out_name, timeout=6.0):
                    stderr = self._read_stderr(self._ec_proc)
                    self._kill_all()
                    return False, f"Echo-cancel failed to start: {stderr}"

                log.info("Echo-cancel ready: %s", ec_out_name)
                final_node = ec_out_name

            # ── Stage 2: Filter-chain (DeepFilterNet) ──
            if self.nc_enabled:
                fc_target = EC_SOURCE_NAME if self.ec_needed else source

                nc = self.config["noise_cancellation"]
                conf = _pw_conf_filter_chain(
                    plugin_path=plugin_path,
                    attenuation_db=nc.get("attenuation_limit_db", 100),
                    min_proc_db=nc.get("min_processing_threshold_db", -15),
                    max_erb_db=nc.get("max_erb_threshold_db", 35),
                    max_df_db=nc.get("max_df_threshold_db", 35),
                    post_filter_beta=nc.get("post_filter_beta", 0.0),
                    target_source=fc_target,
                )
                conf_path = RUNTIME_DIR / "filter-chain.conf"
                conf_path.write_text(conf)

                log.info("Spawning filter-chain process")
                self._fc_proc = subprocess.Popen(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                if not pw_wait_for_node(VIRTUAL_MIC_NAME, timeout=6.0):
                    stderr = self._read_stderr(self._fc_proc)
                    self._kill_all()
                    return False, f"Filter-chain failed to start: {stderr}"

                log.info("Filter-chain ready: %s", VIRTUAL_MIC_NAME)
                final_node = VIRTUAL_MIC_NAME

            # ── Stage 3: Set mic as default ──
            if needs_mic:
                time.sleep(0.3)
                if not pw_set_default_source(final_node):
                    log.warning("Could not set default source to %s", final_node)

            # ── Stage 4: Speaker enhancement ──
            if self.spk_enabled and SPEAKER_CHAIN_CONF.is_file():
                current_sink = pw_get_default_sink()
                if current_sink and not current_sink.startswith("clearvoice"):
                    self.config["previous_default_sink"] = current_sink
                    save_config(self.config)

                log.info("Spawning speaker-chain process")
                self._spk_proc = subprocess.Popen(
                    ["pipewire", "-c", str(SPEAKER_CHAIN_CONF)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )

                if pw_wait_for_node(SPEAKER_SINK_NAME, timeout=6.0):
                    sink_id = pw_find_node_id(SPEAKER_SINK_NAME)
                    if sink_id:
                        pw_set_default_sink(sink_id)
                    log.info("Speaker chain ready: %s", SPEAKER_SINK_NAME)
                else:
                    log.warning("Speaker chain failed to start (non-fatal)")

            self._running = True
            return True, "Pipeline active"

        except Exception as exc:
            log.exception("Pipeline start failed")
            # Restore defaults before killing processes
            prev_src = self.config.get("previous_default_source")
            if prev_src:
                pw_set_default_source(prev_src)
            prev_sink = self.config.get("previous_default_sink")
            if prev_sink:
                sink_id = pw_find_node_id(prev_sink)
                if sink_id:
                    pw_set_default_sink(sink_id)
            self._kill_all()
            return False, str(exc)

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> tuple[bool, str]:
        if not self._running:
            return True, "Already stopped"

        log.info("Stopping pipeline")

        prev = self.config.get("previous_default_source")
        if prev:
            pw_set_default_source(prev)

        prev_sink = self.config.get("previous_default_sink")
        if prev_sink:
            # Restore by name — find its node ID
            sink_id = pw_find_node_id(prev_sink)
            if sink_id:
                pw_set_default_sink(sink_id)

        self._kill_all()
        self._running = False
        return True, "Pipeline stopped"

    def restart(self) -> tuple[bool, str]:
        with self._lock:
            self._stop_locked()
            time.sleep(0.5)
            return self._start_locked()

    # ── Health ──

    def check_health(self) -> bool:
        if not self._running:
            return True
        if self._fc_proc and self._fc_proc.poll() is not None:
            log.error("filter-chain died (rc=%d)", self._fc_proc.returncode)
            return False
        if self._ec_proc and self._ec_proc.poll() is not None:
            log.error("echo-cancel died (rc=%d)", self._ec_proc.returncode)
            return False
        if self._spk_proc and self._spk_proc.poll() is not None:
            log.error("speaker-chain died (rc=%d)", self._spk_proc.returncode)
            return False
        return True

    # ── Internals ──

    @staticmethod
    def _read_stderr(proc: subprocess.Popen, limit: int = 500) -> str:
        try:
            if proc.stderr and proc.poll() is not None:
                return proc.stderr.read(limit).strip()
        except Exception:
            pass
        return "(no output)"

    def _kill_all(self):
        for attr in ("_fc_proc", "_ec_proc", "_spk_proc"):
            proc: subprocess.Popen | None = getattr(self, attr)
            if proc is None:
                continue
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            setattr(self, attr, None)


# ── Tray UI ───────────────────────────────────────────────────────────────────


class ClearVoiceTray:
    """System tray interface."""

    def __init__(self, pipeline: PipelineManager, config: dict):
        self.pipeline = pipeline
        self.config = config

        if HAS_APPINDICATOR:
            self.indicator = AppIndicator.Indicator.new(
                APP_ID,
                ICON_OFF,
                AppIndicator.IndicatorCategory.APPLICATION_STATUS,
            )
            self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
            self.indicator.set_title(APP_NAME)
            self.status_icon = None
        else:
            self.indicator = None
            self.status_icon = Gtk.StatusIcon()
            self.status_icon.set_from_icon_name(ICON_OFF)
            self.status_icon.set_title(APP_NAME)
            self.status_icon.set_visible(True)
            self.status_icon.connect("popup-menu", self._on_popup)

        self._build_menu()
        self._update_icon()
        self._update_status()

        # Health-check timer (every 3 s)
        GLib.timeout_add_seconds(3, self._on_health_tick)

        # Start pipeline if enabled in config (off GTK thread)
        if self.config.get("enabled", True):

            def _deferred_start():
                ok, msg = self.pipeline.start()
                if not ok:
                    log.error("Failed to start pipeline on launch: %s", msg)
                GLib.idle_add(self._update_icon)
                GLib.idle_add(self._update_status)

            threading.Thread(target=_deferred_start, daemon=True).start()

    # ── Menu Construction ──

    def _build_menu(self):
        m = Gtk.Menu()

        # Enable toggle — starts/stops the pipeline and virtual mic
        self._mi_enable = Gtk.CheckMenuItem(label=f"{APP_NAME} Enabled")
        self._mi_enable.set_active(self.config.get("enabled", True))
        self._mi_enable.connect("toggled", self._on_enable)
        m.append(self._mi_enable)

        m.append(Gtk.SeparatorMenuItem())

        # Source selector
        src = Gtk.MenuItem(label="Source")
        self._source_submenu = Gtk.Menu()
        src.set_submenu(self._source_submenu)
        self._source_submenu.connect("show", self._on_source_menu_show)
        m.append(src)

        m.append(Gtk.SeparatorMenuItem())

        # ── Noise Cancellation ──
        self._mi_nc = Gtk.CheckMenuItem(label="Noise Cancellation")
        self._mi_nc.set_active(self.config["noise_cancellation"]["enabled"])
        self._mi_nc.connect("toggled", self._on_nc)
        m.append(self._mi_nc)

        # Attenuation sub
        mi_atten = Gtk.MenuItem(label="    Attenuation")
        sub_atten = Gtk.Menu()
        mi_atten.set_submenu(sub_atten)
        m.append(mi_atten)

        cur_atten = self.config["noise_cancellation"].get("attenuation_limit_db", 100)
        grp = []
        for label, val in [
            ("Light  (40 dB)", 40),
            ("Moderate  (60 dB)", 60),
            ("Strong  (80 dB)", 80),
            ("Maximum  (100 dB)", 100),
        ]:
            ri = Gtk.RadioMenuItem(label=label, group=grp[0] if grp else None)
            ri.set_active(cur_atten == val)
            ri.connect("toggled", self._on_atten, val)
            sub_atten.append(ri)
            grp.append(ri)

        # Advanced NC tunables
        mi_adv = Gtk.MenuItem(label="    Advanced...")
        mi_adv.connect("activate", self._on_nc_advanced)
        m.append(mi_adv)

        m.append(Gtk.SeparatorMenuItem())

        # ── Beamforming ──
        self._mi_bf = Gtk.CheckMenuItem(label="Beamforming")
        self._mi_bf.set_active(self.config["beamforming"]["enabled"])
        self._mi_bf.connect("toggled", self._on_bf)
        m.append(self._mi_bf)

        # Geometry presets sub
        mi_geo = Gtk.MenuItem(label="    Mic Geometry")
        sub_geo = Gtk.Menu()
        mi_geo.set_submenu(sub_geo)
        m.append(mi_geo)

        cur_preset = self.config["beamforming"].get("preset", "laptop-dual-60mm")
        grp2 = []
        for key, preset in MIC_PRESETS.items():
            ri = Gtk.RadioMenuItem(
                label=preset["label"], group=grp2[0] if grp2 else None
            )
            ri.set_active(cur_preset == key)
            ri.connect("toggled", self._on_geo_preset, key)
            sub_geo.append(ri)
            grp2.append(ri)

        sub_geo.append(Gtk.SeparatorMenuItem())
        mi_custom = Gtk.MenuItem(label="Custom...")
        mi_custom.connect("activate", self._on_geo_custom)
        sub_geo.append(mi_custom)

        # ── Echo Cancellation ──
        self._mi_aec = Gtk.CheckMenuItem(label="Echo Cancellation")
        self._mi_aec.set_active(self.config["echo_cancellation"]["enabled"])
        self._mi_aec.connect("toggled", self._on_aec)
        m.append(self._mi_aec)

        m.append(Gtk.SeparatorMenuItem())

        # ── Speaker Enhancement ──
        self._mi_spk = Gtk.CheckMenuItem(label="Speaker Enhancement")
        self._mi_spk.set_active(
            self.config.get("speaker_enhancement", {}).get("enabled", False)
        )
        self._mi_spk.connect("toggled", self._on_spk)
        m.append(self._mi_spk)

        m.append(Gtk.SeparatorMenuItem())

        # Status
        self._mi_status = Gtk.MenuItem(label="Status: Inactive")
        self._mi_status.set_sensitive(False)
        m.append(self._mi_status)

        # Quit
        mi_quit = Gtk.MenuItem(label="Quit")
        mi_quit.connect("activate", self._on_quit)
        m.append(mi_quit)

        m.show_all()

        if self.indicator:
            self.indicator.set_menu(m)
        self.menu = m

    # ── Callbacks ──

    def _on_enable(self, item):
        enabled = item.get_active()
        self.config["enabled"] = enabled
        save_config(self.config)
        if enabled:

            def _do():
                ok, msg = self.pipeline.start()
                GLib.idle_add(self._update_icon)
                GLib.idle_add(self._update_status)
                if not ok:
                    GLib.idle_add(item.set_active, False)
                    GLib.idle_add(self._show_error, msg)

            threading.Thread(target=_do, daemon=True).start()
        else:
            self.pipeline.stop()
            self._update_icon()
            self._update_status()

    def _on_source_menu_show(self, submenu):
        for child in submenu.get_children():
            submenu.remove(child)

        sources = pw_list_sources()
        current = self.config.get("source_device")
        grp = []

        auto = Gtk.RadioMenuItem(label="Auto (system default)", group=None)
        auto.set_active(current is None)
        auto.connect("toggled", self._on_source_pick, None)
        submenu.append(auto)
        grp.append(auto)

        if sources:
            submenu.append(Gtk.SeparatorMenuItem())

        for s in sources:
            ri = Gtk.RadioMenuItem(label=s["description"], group=grp[0])
            ri.set_active(current == s["name"])
            ri.connect("toggled", self._on_source_pick, s["name"])
            submenu.append(ri)
            grp.append(ri)

        submenu.show_all()

    def _on_source_pick(self, item, name):
        if not item.get_active():
            return
        self.config["source_device"] = name
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_nc(self, item):
        self.config["noise_cancellation"]["enabled"] = item.get_active()
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_atten(self, item, val):
        if not item.get_active():
            return
        self.config["noise_cancellation"]["attenuation_limit_db"] = val
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_nc_advanced(self, _item):
        """Dialog for the lesser-used DeepFilterNet controls."""
        nc = self.config["noise_cancellation"]
        dialog = Gtk.Dialog(title="DeepFilterNet Advanced", transient_for=None, flags=0)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK,
            Gtk.ResponseType.OK,
        )
        box = dialog.get_content_area()
        box.set_spacing(6)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        fields = {}
        for label_text, key, lo, hi, default in [
            (
                "Min Processing Threshold (dB)",
                "min_processing_threshold_db",
                -15,
                35,
                -15,
            ),
            ("Max ERB Threshold (dB)", "max_erb_threshold_db", -15, 35, 35),
            ("Max DF Threshold (dB)", "max_df_threshold_db", -15, 35, 35),
            ("Post Filter Beta", "post_filter_beta", 0.0, 0.05, 0.0),
        ]:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            lbl = Gtk.Label(label=label_text)
            lbl.set_xalign(0)
            lbl.set_hexpand(True)
            hbox.pack_start(lbl, True, True, 0)

            adj = Gtk.Adjustment(
                value=nc.get(key, default),
                lower=lo,
                upper=hi,
                step_increment=1 if isinstance(lo, int) else 0.001,
                page_increment=5 if isinstance(lo, int) else 0.01,
            )
            spin = Gtk.SpinButton(
                adjustment=adj, digits=0 if isinstance(lo, int) else 3
            )
            hbox.pack_end(spin, False, False, 0)
            box.add(hbox)
            fields[key] = spin

        dialog.show_all()
        if dialog.run() == Gtk.ResponseType.OK:
            for key, spin in fields.items():
                nc[key] = spin.get_value()
            save_config(self.config)
            if self.pipeline.running:
                self._async_restart()
        dialog.destroy()

    def _on_bf(self, item):
        self.config["beamforming"]["enabled"] = item.get_active()
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_geo_preset(self, item, key):
        if not item.get_active():
            return
        self.config["beamforming"]["preset"] = key
        self.config["beamforming"]["custom_geometry"] = None
        save_config(self.config)
        if self.pipeline.running and self.pipeline.bf_enabled:
            self._async_restart()

    def _on_geo_custom(self, _item):
        dialog = Gtk.Dialog(title="Custom Mic Geometry", transient_for=None, flags=0)
        dialog.add_buttons(
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK,
            Gtk.ResponseType.OK,
        )
        box = dialog.get_content_area()
        box.set_spacing(8)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        lbl = Gtk.Label(
            label=(
                "Mic coordinates in meters, comma-separated:\n"
                "  x1,y1,z1,x2,y2,z2,...\n\n"
                "Example (2 mics, 60mm apart):\n"
                "  -0.03,0,0,0.03,0,0"
            )
        )
        lbl.set_xalign(0)
        box.add(lbl)

        entry = Gtk.Entry()
        entry.set_placeholder_text("-0.03,0,0,0.03,0,0")
        cur = self.config["beamforming"].get("custom_geometry", "")
        if cur:
            entry.set_text(cur)
        box.add(entry)

        dialog.show_all()
        if dialog.run() == Gtk.ResponseType.OK:
            text = entry.get_text().strip()
            if text:
                # Validate: must be comma-separated floats, count divisible by 3
                try:
                    vals = [float(v) for v in text.split(",")]
                    if len(vals) < 3 or len(vals) % 3 != 0:
                        raise ValueError("Need 3 coords per mic (x,y,z)")
                except ValueError as exc:
                    self._show_error(f"Invalid geometry: {exc}")
                    dialog.destroy()
                    return
                self.config["beamforming"]["custom_geometry"] = text
                self.config["beamforming"]["preset"] = "custom"
                save_config(self.config)
                if self.pipeline.running and self.pipeline.bf_enabled:
                    self._async_restart()
        dialog.destroy()

    def _on_aec(self, item):
        self.config["echo_cancellation"]["enabled"] = item.get_active()
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_spk(self, item):
        if "speaker_enhancement" not in self.config:
            self.config["speaker_enhancement"] = {}
        self.config["speaker_enhancement"]["enabled"] = item.get_active()
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_quit(self, _item):
        self.pipeline.stop()
        save_config(self.config)
        Gtk.main_quit()

    def _on_popup(self, icon, button, timestamp):
        self.menu.popup(
            None, None, Gtk.StatusIcon.position_menu, icon, button, timestamp
        )

    # ── Helpers ──

    def _async_restart(self):
        """Restart the pipeline off the GTK thread."""

        def _do():
            ok, msg = self.pipeline.restart()
            GLib.idle_add(self._update_icon)
            GLib.idle_add(self._update_status)
            if not ok:
                GLib.idle_add(self._show_error, msg)
                GLib.idle_add(self._mi_enable.set_active, False)

        threading.Thread(target=_do, daemon=True).start()

    def _update_icon(self, nodes_active: bool = False):
        if not self.pipeline.running:
            icon = ICON_OFF
        elif nodes_active:
            icon = ICON_ACTIVE
        else:
            icon = ICON_STANDBY
        if self.indicator:
            self.indicator.set_icon_full(icon, APP_NAME)
        elif self.status_icon:
            self.status_icon.set_from_icon_name(icon)

    def _update_status(self):
        if self.pipeline.running:
            parts = []
            if self.pipeline.nc_enabled:
                parts.append("NC")
            if self.pipeline.bf_enabled:
                parts.append("BF")
            if self.pipeline.aec_enabled:
                parts.append("AEC")
            if self.pipeline.spk_enabled:
                parts.append("SPK")
            tag = "+".join(parts) or "active"
            src = self.config.get("source_device") or "auto"
            self._mi_status.set_label(f"Status: Active [{tag}] src={src}")
        else:
            self._mi_status.set_label("Status: Off")

    def _on_health_tick(self):
        if self.pipeline.running:
            if not self.pipeline.check_health():
                log.warning("Health check failed — restarting pipeline")
                self._async_restart()
                return True
            # Poll node state for icon update (lightweight)
            active = pw_nodes_active()
            self._update_icon(nodes_active=active)
        return True  # keep timer

    @staticmethod
    def _show_error(msg: str):
        d = Gtk.MessageDialog(
            transient_for=None,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=f"{APP_NAME} Error",
        )
        d.format_secondary_text(str(msg))
        d.run()
        d.destroy()


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(),
        ],
    )

    log.info("%s %s starting", APP_NAME, VERSION)

    # Dependency check
    missing = check_dependencies()
    if missing:
        msg = "Missing dependencies:\n" + "\n".join(f"  - {m}" for m in missing)
        log.error(msg)
        try:
            d = Gtk.MessageDialog(
                transient_for=None,
                flags=0,
                message_type=Gtk.MessageType.ERROR,
                buttons=Gtk.ButtonsType.OK,
                text=f"{APP_NAME} — Missing Dependencies",
            )
            d.format_secondary_text(msg)
            d.run()
            d.destroy()
        except Exception:
            print(msg, file=sys.stderr)
        sys.exit(1)

    config = load_config()
    pipeline = PipelineManager(config)

    # Tray — must be created before entering GTK main loop
    tray = ClearVoiceTray(pipeline, config)

    # Clean shutdown on signals
    def _shutdown(*_args):
        pipeline.stop()
        save_config(config)
        Gtk.main_quit()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _shutdown)

    log.info("Tray ready — entering GTK main loop")
    Gtk.main()

    # Belt-and-suspenders cleanup
    pipeline.stop()
    save_config(config)
    log.info("%s shutdown complete", APP_NAME)


if __name__ == "__main__":
    main()
