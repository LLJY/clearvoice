#!/usr/bin/env python3
"""ClearVoice — PipeWire noise cancellation, beamforming & AEC system tray tool.

Creates a virtual microphone with DeepFilterNet noise cancellation,
WebRTC-based beamforming, and acoustic echo cancellation via PipeWire.
"""

import atexit
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
PIDFILE = RUNTIME_DIR / "clearvoice.pid"

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
ICON_ACTIVE = "audio-input-microphone-high"  # full bars — processing audio
ICON_STANDBY = "audio-input-microphone-low"  # low bar — enabled, idle
ICON_OFF = "microphone-sensitivity-muted-symbolic"  # slashed — disabled

log = logging.getLogger(APP_ID)


# ── Mic Geometry Presets ──────────────────────────────────────────────────────

MIC_PRESETS = {
    "laptop-dual-50mm": {
        "label": "Dual 50mm (Aftershock)",
        "geometry": "-0.025,0,0,0.025,0,0",
    },
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
    "input_gain_percent": 70,
    "output_gain_percent": 100,
    "speaker_gain_percent": 80,
    "headphone_gain_percent": 100,
    "lock_base_mic_audio": True,
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
        "preset": "laptop-dual-50mm",
        "custom_geometry": None,
    },
    "echo_cancellation": {
        "enabled": False,
    },
    "speaker_enhancement": {
        "enabled": True,
    },
    "studio_voice": {
        "enabled": True,
    },
    "previous_default_source": None,
    "previous_default_sink": None,
}


# ── Config I/O ────────────────────────────────────────────────────────────────


def _clamp_percent(value, default: int) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default


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
    for key in (
        "input_gain_percent",
        "output_gain_percent",
        "speaker_gain_percent",
        "headphone_gain_percent",
    ):
        config[key] = _clamp_percent(config.get(key), DEFAULT_CONFIG[key])
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
        result = subprocess.run(
            ["pw-dump"],
            capture_output=True,
            text=True,
            timeout=5,
            env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"},
        )
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


def pactl_list_sinks() -> list[dict] | None:
    """Return PulseAudio-compatible sink data, or None on query errors."""
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sinks"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            log.warning("Could not list sinks: %s", result.stderr.strip())
            return None
        sinks = json.loads(result.stdout)
        if not isinstance(sinks, list) or not all(
            isinstance(sink, dict) for sink in sinks
        ):
            log.warning("Unexpected sink list response")
            return None
        return sinks
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log.warning("Could not list sinks: %s", exc)
        return None


def pw_get_sink_active_port(sink_name: str) -> str | None:
    """Return the active port for exactly *sink_name*, if known."""
    sinks = pactl_list_sinks()
    if sinks is None:
        return None
    for sink in sinks:
        if sink.get("name") == sink_name:
            active_port = sink.get("active_port")
            return active_port if isinstance(active_port, str) else None
    return None


def pw_move_playback_streams(physical_sink: str, new_sink: str) -> bool:
    """Move normal playback streams between ClearVoice's managed sinks."""
    sinks = pactl_list_sinks()
    if sinks is None:
        return False
    sink_names = {
        str(sink.get("index")): sink.get("name")
        for sink in sinks
        if sink.get("name") in (physical_sink, SPEAKER_SINK_NAME)
        and sink.get("index") is not None
    }
    if not sink_names:
        return False
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", "sink-inputs"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            log.warning("Could not list playback streams: %s", result.stderr.strip())
            return False
        sink_inputs = json.loads(result.stdout)
        if not isinstance(sink_inputs, list):
            log.warning("Unexpected playback stream list response")
            return False
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log.warning("Could not list playback streams: %s", exc)
        return False

    success = True
    for sink_input in sink_inputs:
        if not isinstance(sink_input, dict):
            continue
        current_sink = sink_names.get(str(sink_input.get("sink")))
        properties = sink_input.get("properties", {})
        node_name = properties.get("node.name", "") if isinstance(properties, dict) else ""
        if not current_sink or current_sink == new_sink or str(node_name).startswith("clearvoice_"):
            continue
        index = sink_input.get("index")
        if index is None:
            continue
        try:
            result = subprocess.run(
                ["pactl", "move-sink-input", str(index), new_sink],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode != 0:
                log.warning(
                    "Could not move playback stream %s to %s: %s",
                    index,
                    new_sink,
                    result.stderr.strip(),
                )
                success = False
        except (OSError, subprocess.TimeoutExpired) as exc:
            log.warning("Could not move playback stream %s to %s: %s", index, new_sink, exc)
            success = False
    return success


def pw_set_default_sink(node_id: int) -> bool:
    """Set default sink by PipeWire node ID via wpctl."""
    try:
        r = subprocess.run(
            ["wpctl", "set-default", str(node_id)],
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"},
        )
        return r.returncode == 0
    except Exception:
        return False


def pw_find_node_id(node_name: str, manager: bool = False) -> int | None:
    """Find a PipeWire node ID by node.name."""
    try:
        r = subprocess.run(
            ["pw-dump"],
            capture_output=True,
            text=True,
            timeout=5,
            env=(
                {**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"}
                if manager
                else None
            ),
        )
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


def pw_set_node_volume(node_name: str, percent: int) -> bool:
    """Set a node volume through WirePlumber's manager-visible remote."""
    node_id = pw_find_node_id(node_name, manager=True)
    if node_id is None:
        log.warning("Could not find node %s to set volume", node_name)
        return False
    percent = _clamp_percent(percent, 100)
    try:
        r = subprocess.run(
            ["wpctl", "set-volume", str(node_id), f"{percent / 100:.2f}"],
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"},
        )
        if r.returncode != 0:
            log.warning("Could not set volume for %s: %s", node_name, r.stderr.strip())
            return False
        return True
    except Exception as exc:
        log.warning("Could not set volume for %s: %s", node_name, exc)
        return False


def wp_set_setting(key: str, value: str) -> bool:
    """Set an optional dynamic WirePlumber policy setting without blocking startup."""
    try:
        r = subprocess.run(
            ["wpctl", "settings", "--save", key, value],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if r.returncode != 0:
            log.warning("Could not set WirePlumber setting %s: %s", key, r.stderr.strip())
            return False
        return True
    except Exception as exc:
        log.warning("Could not set WirePlumber setting %s: %s", key, exc)
        return False


class PipeWireMonitor:
    """Event-driven PipeWire node state monitor via ``pw-dump --monitor``.

    Fires *on_state_change(bool)* when any clearvoice node transitions
    to/from 'running'. Zero CPU when nothing changes.
    """

    def __init__(self, on_state_change: callable):
        self.on_state_change = on_state_change
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._active = False
        self._enabled = False

    @property
    def nodes_active(self) -> bool:
        return self._active

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._enabled = True
        self._proc = subprocess.Popen(
            ["pw-dump", "--monitor", "--no-colors"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,  # unbuffered
        )
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        log.info("PipeWire monitor started")

    def stop(self):
        self._enabled = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _read_loop(self):
        """Read pw-dump --monitor output, parse JSON chunks by bracket depth."""
        try:
            buf = b""
            depth = 0
            initial_done = False
            while self._enabled:
                line = self._proc.stdout.readline()
                if not line:
                    break  # EOF
                buf += line
                depth += line.count(b"[") + line.count(b"{")
                depth -= line.count(b"]") + line.count(b"}")
                if depth == 0 and buf.strip():
                    if initial_done:
                        # Only parse diff events, skip initial dump
                        self._check_diff(buf)
                    else:
                        # Initial dump done — check starting state
                        self._check_diff(buf)
                        initial_done = True
                    buf = b""
        except Exception as exc:
            log.debug("PipeWire monitor read error: %s", exc)

    def _check_diff(self, raw: bytes):
        """Scan a JSON chunk for clearvoice node state changes."""
        try:
            objects = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(objects, list):
            return

        now_active = False
        found_ours = False
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            info = obj.get("info")
            if not isinstance(info, dict):
                continue
            props = info.get("props")
            if not isinstance(props, dict):
                continue
            name = props.get("node.name", "")
            if not name.startswith("clearvoice"):
                continue
            found_ours = True
            if info.get("state") == "running":
                now_active = True
                break

        if found_ours and now_active != self._active:
            self._active = now_active
            GLib.idle_add(self.on_state_change, now_active)


def pw_set_default_source(name: str) -> bool:
    node_id = pw_find_node_id(name, manager=True)
    if node_id is None:
        return False
    try:
        r = subprocess.run(
            ["wpctl", "set-default", str(node_id)],
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"},
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
    studio_voice: bool = True,
) -> str:
    """Build a PipeWire config that loads a DeepFilterNet filter-chain."""
    target_line = ""
    if target_source:
        target_line = f'target.object = "{target_source}"'

    studio_pre_nodes = ""
    studio_post_nodes = ""
    if studio_voice:
        studio_pre_nodes = (
            "                    { type = builtin name = hpf label = bq_highpass\n"
            '                        control = { "Freq" = 70.0 "Q" = 0.707 } }\n'
            "                    { type = builtin name = body label = bq_peaking\n"
            '                        control = { "Freq" = 150.0 "Q" = 0.8 "Gain" = 1.5 } }\n'
            "                    { type = builtin name = lowmid label = bq_peaking\n"
            '                        control = { "Freq" = 350.0 "Q" = 1.0 "Gain" = -1.0 } }\n'
        )
        studio_post_nodes = (
            "                    {\n"
            "                        type   = lv2\n"
            "                        name   = deesser\n"
            '                        plugin = "http://calf.sourceforge.net/plugins/Deesser"\n'
            "                        control = {\n"
            '                            "bypass"    = 0\n'
            '                            "detection" = 0\n'
            '                            "mode"      = 1\n'
            '                            "threshold" = 0.18\n'
            '                            "ratio"     = 3.0\n'
            '                            "laxity"    = 15\n'
            '                            "makeup"    = 1.0\n'
            '                            "f1_freq"   = 6000.0\n'
            '                            "f2_freq"   = 6000.0\n'
            "                        }\n"
            "                    }\n"
            "                    {\n"
            "                        type   = lv2\n"
            "                        name   = limiter\n"
            '                        plugin = "http://calf.sourceforge.net/plugins/Limiter"\n'
            "                        control = {\n"
            '                            "bypass"       = 0\n'
            '                            "level_in"     = 1.0\n'
            '                            "level_out"    = 1.0\n'
            '                            "limit"        = 0.891251\n'
            '                            "attack"       = 0.5\n'
            '                            "release"      = 50.0\n'
            '                            "asc"          = 1\n'
            '                            "asc_coeff"    = 0.5\n'
            '                            "oversampling" = 1\n'
            '                            "auto_level"   = 0\n'
            "                        }\n"
            "                    }\n"
        )

    if studio_voice:
        links = (
            '                    { output = "pretrim:Out" input = "deepfilter:Audio In" }\n'
            '                    { output = "deepfilter:Audio Out" input = "restore:In" }\n'
            '                    { output = "restore:Out" input = "hpf:In" }\n'
            '                    { output = "hpf:Out" input = "body:In" }\n'
            '                    { output = "body:Out" input = "lowmid:In" }\n'
            '                    { output = "lowmid:Out" input = "agc:in_l" }\n'
            '                    { output = "lowmid:Out" input = "agc:in_r" }\n'
            '                    { output = "agc:out_l" input = "deesser:in_l" }\n'
            '                    { output = "agc:out_r" input = "deesser:in_r" }\n'
            '                    { output = "deesser:out_l" input = "limiter:in_l" }\n'
            '                    { output = "deesser:out_r" input = "limiter:in_r" }\n'
        )
    else:
        links = (
            '                    { output = "pretrim:Out" input = "deepfilter:Audio In" }\n'
            '                    { output = "deepfilter:Audio Out" input = "restore:In" }\n'
            '                    { output = "restore:Out" input = "agc:in_l" }\n'
            '                    { output = "restore:Out" input = "agc:in_r" }\n'
        )

    return (
        "# ClearVoice filter-chain (auto-generated)\n"
        "context.properties = {\n"
        "    log.level = 0\n"
        '    application.name = "ClearVoice"\n'
        '    application.id = "org.clearvoice.ClearVoice"\n'
        "    clearvoice.client = true\n"
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
        "                    { type = builtin name = pretrim label = linear\n"
        '                        control = { "Mult" = 0.630957344 "Add" = 0.0 } }\n'
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
        "                    { type = builtin name = restore label = linear\n"
        '                        control = { "Mult" = 1.584893192 "Add" = 0.0 } }\n'
        f"{studio_pre_nodes}"
        "                    # Calibrated output leveling\n"
        "                    {\n"
        "                        type   = lv2\n"
        "                        name   = agc\n"
        '                        plugin = "http://calf.sourceforge.net/plugins/Compressor"\n'
        "                        control = {\n"
        '                            "bypass"      = 0\n'
        '                            "level_in"    = 1.0\n'
        '                            "threshold"   = 0.15\n'  # ~-16dB, catches quiet speech
        '                            "ratio"       = 4.0\n'  # strong compression
        '                            "attack"      = 10.0\n'  # 10ms, catches syllables
        '                            "release"     = 150.0\n'  # 150ms, smooth
        '                            "makeup"      = 2.5\n'  # bring compressed signal up
        '                            "knee"        = 4.0\n'  # soft knee for natural sound
        '                            "detection"   = 0\n'  # RMS detection
        '                            "stereo_link" = 1\n'
        '                            "mix"         = 1.0\n'  # 100% wet
        "                        }\n"
        "                    }\n"
        f"{studio_post_nodes}"
        "                ]\n"
        "                links = [\n"
        f"{links}"
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
        "                state.restore-props = false\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "]\n"
    )


def _pw_conf_echo_cancel(
    target_source: str | None = None,
    monitor_mode: bool = False,
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
    restore_props = "state.restore-props = false" if not is_intermediate else ""

    return (
        "# ClearVoice echo-cancel (auto-generated)\n"
        "context.properties = {\n"
        "    log.level = 0\n"
        '    application.name = "ClearVoice"\n'
        '    application.id = "org.clearvoice.ClearVoice"\n'
        "    clearvoice.client = true\n"
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
        f"            monitor.mode = {'true' if monitor_mode else 'false'}\n"
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
        f"                {restore_props}\n"
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
        self._transitioning = False  # True during stop/start — suppresses health checks
        self._lock = threading.Lock()
        self._base_mic_node: str | None = None
        self._physical_sink: str | None = None
        self._headphone_mode = False
        self._route_confirmed = False

        # Ensure child processes are cleaned up if we crash
        atexit.register(self._kill_all)

    # ── Properties ──

    @property
    def running(self) -> bool:
        return self._running

    @property
    def transitioning(self) -> bool:
        return self._transitioning

    @property
    def headphone_mode(self) -> bool:
        return self._headphone_mode

    @property
    def nc_enabled(self) -> bool:
        return self.config["noise_cancellation"]["enabled"]

    @property
    def bf_enabled(self) -> bool:
        return self.config["beamforming"]["enabled"] and not self._headphone_mode

    @property
    def aec_enabled(self) -> bool:
        return self.config["echo_cancellation"]["enabled"] and not self._headphone_mode

    @property
    def ec_needed(self) -> bool:
        return self.bf_enabled or self.aec_enabled

    @property
    def spk_enabled(self) -> bool:
        return (
            self.config.get("speaker_enhancement", {}).get("enabled", False)
            and not self._headphone_mode
        )

    @property
    def studio_enabled(self) -> bool:
        return self.config.get("studio_voice", {}).get("enabled", True)

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

    # ── Output Route Resolution ──

    def _resolve_physical_sink(self) -> str | None:
        """Use the original non-ClearVoice sink as the route authority."""
        sinks = pactl_list_sinks()
        sink_names = {sink.get("name") for sink in sinks or []}
        previous = self.config.get("previous_default_sink")
        if (
            isinstance(previous, str)
            and previous
            and not previous.startswith("clearvoice")
            and (sinks is None or previous in sink_names)
        ):
            return previous

        current = pw_get_default_sink()
        if current and not current.startswith("clearvoice"):
            self.config["previous_default_sink"] = current
            save_config(self.config)
            return current
        return None

    def detect_headphone_mode(self) -> bool | None:
        """Return the currently confirmed route without changing stored mode."""
        physical_sink = self._physical_sink or self._resolve_physical_sink()
        if not physical_sink:
            return None
        active_port = pw_get_sink_active_port(physical_sink)
        if active_port == "analog-output-headphones":
            return True
        if active_port == "analog-output-speaker":
            return False
        return None

    def _refresh_headphone_mode(self):
        mode = self.detect_headphone_mode()
        if mode is None:
            if not self._route_confirmed:
                log.warning("Could not determine output route; defaulting to speaker mode")
            return
        self._headphone_mode = mode
        self._route_confirmed = True

    # ── Gain / Policy Management ──

    @staticmethod
    def _publish_lock_state(locked: bool) -> bool:
        return wp_set_setting(
            "clearvoice.lock-base-mic-audio", str(bool(locked)).lower()
        )

    @staticmethod
    def _publish_base_mic_node(node_name: str) -> bool:
        return wp_set_setting("clearvoice.base-mic-node", json.dumps(node_name))

    @staticmethod
    def _publish_input_gain(percent: int) -> bool:
        return wp_set_setting("clearvoice.base-mic-gain", f"{percent / 100:.2f}")

    def set_input_gain(self, percent: int) -> bool:
        percent = _clamp_percent(percent, DEFAULT_CONFIG["input_gain_percent"])
        self.config["input_gain_percent"] = percent
        policy_set = self._publish_input_gain(percent)
        source = self._base_mic_node or self._resolve_source()
        if source:
            self._base_mic_node = source
            volume_set = pw_set_node_volume(source, percent)
            return policy_set and volume_set
        return False

    def set_output_gain(self, percent: int) -> bool:
        percent = _clamp_percent(percent, DEFAULT_CONFIG["output_gain_percent"])
        self.config["output_gain_percent"] = percent
        if pw_node_exists(VIRTUAL_MIC_NAME):
            return pw_set_node_volume(VIRTUAL_MIC_NAME, percent)
        return False

    def set_route_gains(self, speaker_percent: int, headphone_percent: int) -> bool:
        """Save and immediately apply the gain for the active output route."""
        speaker_percent = _clamp_percent(
            speaker_percent, DEFAULT_CONFIG["speaker_gain_percent"]
        )
        headphone_percent = _clamp_percent(
            headphone_percent, DEFAULT_CONFIG["headphone_gain_percent"]
        )
        self.config["speaker_gain_percent"] = speaker_percent
        self.config["headphone_gain_percent"] = headphone_percent
        physical_sink = self._physical_sink or self._resolve_physical_sink()
        if not physical_sink:
            return False
        if self._headphone_mode:
            return pw_set_node_volume(physical_sink, headphone_percent)
        if self.spk_enabled and pw_node_exists(SPEAKER_SINK_NAME):
            return pw_set_node_volume(SPEAKER_SINK_NAME, speaker_percent)
        return pw_set_node_volume(physical_sink, speaker_percent)

    def _set_playback_sink(self, sink_name: str, gain_percent: int) -> bool:
        """Set the sink volume and default, then move non-ClearVoice streams."""
        gain_set = pw_set_node_volume(sink_name, gain_percent)
        sink_id = pw_find_node_id(sink_name)
        if sink_id is None:
            log.warning("Could not find playback sink %s", sink_name)
            return False
        default_set = pw_set_default_sink(sink_id)
        if not default_set:
            log.warning("Could not set default playback sink to %s", sink_name)
            return False
        if self._physical_sink:
            pw_move_playback_streams(self._physical_sink, sink_name)
        return gain_set

    def _start_playback_output(self):
        """Activate the appropriate speaker or headphone playback route."""
        physical_sink = self._physical_sink
        if not physical_sink:
            log.warning("No physical playback sink found; leaving playback route unchanged")
            return
        if self._headphone_mode:
            self._set_playback_sink(
                physical_sink, self.config["headphone_gain_percent"]
            )
            return
        if not self.spk_enabled or not SPEAKER_CHAIN_CONF.is_file():
            if not self.spk_enabled:
                self._set_playback_sink(
                    physical_sink, self.config["speaker_gain_percent"]
                )
            else:
                log.warning("Speaker chain config not found; using physical speakers")
                self._set_playback_sink(
                    physical_sink, self.config["speaker_gain_percent"]
                )
            return

        pw_set_node_volume(physical_sink, 100)
        log.info("Spawning speaker-chain process")
        spk_log = open(RUNTIME_DIR / "speaker-chain.log", "w")
        self._spk_proc = subprocess.Popen(
            ["pipewire", "-c", str(SPEAKER_CHAIN_CONF)],
            stdout=subprocess.DEVNULL,
            stderr=spk_log,
        )
        if pw_wait_for_node(SPEAKER_SINK_NAME, timeout=6.0):
            self._set_playback_sink(
                SPEAKER_SINK_NAME, self.config["speaker_gain_percent"]
            )
            log.info("Speaker chain ready: %s", SPEAKER_SINK_NAME)
            return
        log.warning("Speaker chain failed to start; using physical speakers")
        self._set_playback_sink(physical_sink, self.config["speaker_gain_percent"])

    def set_lock_base_mic_audio(self, locked: bool) -> bool:
        self.config["lock_base_mic_audio"] = bool(locked)
        if self._running:
            return self._publish_lock_state(self.config["lock_base_mic_audio"])
        return True

    # ── Start / Stop ──

    def start(self) -> tuple[bool, str]:
        with self._lock:
            try:
                return self._start_locked()
            except Exception as exc:
                log.exception("Pipeline start failed")
                return self._fail_start(str(exc))

    @staticmethod
    def _cleanup_orphans():
        """Kill any orphaned ClearVoice PipeWire processes from a previous crash."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "pipewire -c.*/clearvoice/"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            pids = result.stdout.strip().split()
            if pids:
                log.warning(
                    "Cleaning up %d orphaned PipeWire processes: %s", len(pids), pids
                )
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                    except (ProcessLookupError, ValueError):
                        pass
                time.sleep(0.5)
        except Exception:
            pass

    def _start_locked(self) -> tuple[bool, str]:
        if self._running:
            return True, "Already running"

        self._running = False
        self._base_mic_node = None
        self._physical_sink = self._resolve_physical_sink()
        self._refresh_headphone_mode()

        # Fail open even if an earlier start attempt did not reach the final stage.
        self._publish_lock_state(False)

        if not self.any_processing:
            return False, "Enable at least one processing feature"

        # Clean up orphans only on first start (not restarts)
        if not hasattr(self, "_started_once"):
            self._cleanup_orphans()
            self._started_once = True

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
            "Starting pipeline — source=%s mic=%s spk=%s hp=%s",
            source,
            needs_mic,
            self.spk_enabled,
            self._headphone_mode,
        )

        # Remember current defaults so we can restore them
        if needs_mic:
            current_default = pw_get_default_source()
            if current_default and not current_default.startswith("clearvoice"):
                self.config["previous_default_source"] = current_default
                save_config(self.config)

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

        # Make the eventual AEC monitor use the selected playback route.
        self._start_playback_output()

        if needs_mic:
            self._base_mic_node = source
            self._publish_lock_state(False)
            base_node_ready = self._publish_base_mic_node(source)
            input_gain_ready = self.set_input_gain(
                self.config["input_gain_percent"]
            )
            base_mic_ready = base_node_ready and input_gain_ready
            if self.config["lock_base_mic_audio"] and not base_mic_ready:
                return self._fail_start("Could not prepare base mic audio lock")

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
                    monitor_mode=self.aec_enabled,
                    beamforming=self.bf_enabled,
                    mic_geometry=geometry,
                    source_name=ec_out_name,
                    source_desc=ec_out_desc,
                    is_intermediate=self.nc_enabled,
                )
                conf_path = RUNTIME_DIR / "echo-cancel.conf"
                conf_path.write_text(conf)

                log.info("Spawning echo-cancel process")
                ec_log = open(RUNTIME_DIR / "echo-cancel.log", "w")
                self._ec_proc = subprocess.Popen(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=ec_log,
                )

                if not pw_wait_for_node(ec_out_name, timeout=6.0):
                    ec_log.flush()
                    stderr = (RUNTIME_DIR / "echo-cancel.log").read_text()[-500:]
                    return self._fail_start(f"Echo-cancel failed to start: {stderr}")

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
                    studio_voice=self.studio_enabled,
                )
                conf_path = RUNTIME_DIR / "filter-chain.conf"
                conf_path.write_text(conf)

                log.info("Spawning filter-chain process")
                fc_log = open(RUNTIME_DIR / "filter-chain.log", "w")
                self._fc_proc = subprocess.Popen(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=fc_log,
                )

                if not pw_wait_for_node(VIRTUAL_MIC_NAME, timeout=6.0):
                    fc_log.flush()
                    stderr = (RUNTIME_DIR / "filter-chain.log").read_text()[-500:]
                    return self._fail_start(f"Filter-chain failed to start: {stderr}")

                log.info("Filter-chain ready: %s", VIRTUAL_MIC_NAME)
                final_node = VIRTUAL_MIC_NAME

            # ── Stage 3: Set mic as default + configured output gain ──
            if needs_mic:
                time.sleep(0.3)
                output_gain_ready = self.set_output_gain(
                    self.config["output_gain_percent"]
                )
                if not pw_set_default_source(final_node):
                    return self._fail_start(
                        f"Could not set default source to {final_node}"
                    )
                if self.config["lock_base_mic_audio"] and not output_gain_ready:
                    return self._fail_start("Could not prepare ClearVoice output gain lock")

            lock_state = self.config["lock_base_mic_audio"] if needs_mic else False
            if not self._publish_lock_state(lock_state):
                return self._fail_start("Could not publish base mic audio lock policy")

            self._running = True
            return True, "Pipeline active"

        except Exception as exc:
            log.exception("Pipeline start failed")
            return self._fail_start(str(exc))

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self) -> tuple[bool, str]:
        if not self._running:
            return True, "Already stopped"

        was_transitioning = self._transitioning
        self._transitioning = True
        log.info("Stopping pipeline")

        self._publish_lock_state(False)
        self._kill_all()
        self._restore_previous_defaults()
        self._base_mic_node = None
        self._running = False
        if not was_transitioning:
            self._transitioning = False
        return True, "Pipeline stopped"

    def _fail_start(self, msg: str) -> tuple[bool, str]:
        self._running = False
        self._publish_lock_state(False)
        self._kill_all()
        self._restore_previous_defaults()
        self._base_mic_node = None
        return False, msg

    def _restore_previous_defaults(self):
        prev = self.config.get("previous_default_source")
        if prev:
            pw_set_default_source(prev)

        prev_sink = self.config.get("previous_default_sink")
        if prev_sink:
            # Restore by name — find its node ID
            sink_id = pw_find_node_id(prev_sink)
            if sink_id:
                pw_set_default_sink(sink_id)

    def restart(self) -> tuple[bool, str]:
        with self._lock:
            self._transitioning = True
            try:
                self._stop_locked()
                time.sleep(0.5)
                return self._start_locked()
            except Exception as exc:
                log.exception("Pipeline restart failed")
                return self._fail_start(str(exc))
            finally:
                self._transitioning = False

    # ── Health ──

    def check_health(self) -> bool:
        if not self._running or self._transitioning:
            return True
        dead = []
        if self._fc_proc and self._fc_proc.poll() is not None:
            dead.append(("filter-chain", self._fc_proc.returncode))
        if self._ec_proc and self._ec_proc.poll() is not None:
            dead.append(("echo-cancel", self._ec_proc.returncode))
        if self._spk_proc and self._spk_proc.poll() is not None:
            dead.append(("speaker-chain", self._spk_proc.returncode))
        if dead:
            for name, rc in dead:
                log.error("%s died (rc=%d)", name, rc)
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
        # Send SIGTERM to all processes first (non-blocking)
        procs = []
        for attr in ("_fc_proc", "_ec_proc", "_spk_proc"):
            proc: subprocess.Popen | None = getattr(self, attr)
            if proc is not None and proc.poll() is None:
                proc.terminate()
                procs.append((attr, proc))
            else:
                setattr(self, attr, None)

        # Wait for all in parallel with a single deadline
        deadline = time.monotonic() + 2.0
        for attr, proc in procs:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    log.warning("Process %s (pid %d) did not die", attr, proc.pid)
            setattr(self, attr, None)


# ── Tray UI ───────────────────────────────────────────────────────────────────


class ClearVoiceTray:
    """System tray interface."""

    def __init__(self, pipeline: PipelineManager, config: dict):
        self.pipeline = pipeline
        self.config = config
        self._pw_monitor: PipeWireMonitor | None = None
        self._route_probe_pending = False
        self._route_restart_pending = False
        self._quitting = False

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

        # Process health check (10s)
        GLib.timeout_add_seconds(10, self._on_health_tick)

        # Jack-route probes run off the GTK thread.
        GLib.timeout_add_seconds(2, self._on_route_tick)

        # Event-driven node state monitor
        self._pw_monitor = PipeWireMonitor(on_state_change=self._on_pw_state_change)
        self._pw_monitor.start()

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

        self._mi_lock_base_mic = Gtk.CheckMenuItem(label="Lock Base Mic Audio")
        self._mi_lock_base_mic.set_active(self.config["lock_base_mic_audio"])
        self._mi_lock_base_mic.connect("toggled", self._on_lock_base_mic)
        m.append(self._mi_lock_base_mic)

        mi_gains = Gtk.MenuItem(label="Audio Gains…")
        mi_gains.connect("activate", self._on_audio_gains)
        m.append(mi_gains)

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

        self._mi_studio = Gtk.CheckMenuItem(label="Studio Voice")
        self._mi_studio.set_active(self.config["studio_voice"]["enabled"])
        self._mi_studio.connect("toggled", self._on_studio_voice)
        m.append(self._mi_studio)

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
            self._route_restart_pending = False
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

    def _on_lock_base_mic(self, item):
        locked = item.get_active()
        previous = self.config["lock_base_mic_audio"]
        if locked == previous:
            return
        self.config["lock_base_mic_audio"] = locked
        save_config(self.config)
        item.set_sensitive(False)

        def _do():
            if self.pipeline.set_lock_base_mic_audio(locked):
                GLib.idle_add(item.set_sensitive, True)
                return

            def _restore():
                self.config["lock_base_mic_audio"] = previous
                save_config(self.config)
                item.set_active(previous)
                item.set_sensitive(True)
                self._show_error("Could not update base mic audio lock policy")

            GLib.idle_add(_restore)

        threading.Thread(target=_do, daemon=True).start()

    def _on_audio_gains(self, _item):
        dialog = Gtk.Dialog(title="Audio Gains", transient_for=None, flags=0)
        dialog.set_default_size(720, -1)
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

        scales = {}
        for label_text, key in [
            ("Base Mic Gain", "input_gain_percent"),
            ("ClearVoice Mic Output Gain", "output_gain_percent"),
            ("Speaker Gain", "speaker_gain_percent"),
            ("Headphone Gain", "headphone_gain_percent"),
        ]:
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            label = Gtk.Label(label=label_text)
            label.set_xalign(0)
            hbox.pack_start(label, False, False, 0)
            scale = Gtk.Scale(
                orientation=Gtk.Orientation.HORIZONTAL,
                adjustment=Gtk.Adjustment(
                    value=self.config[key],
                    lower=0,
                    upper=100,
                    step_increment=1,
                    page_increment=10,
                ),
            )
            scale.set_digits(0)
            scale.set_draw_value(True)
            scale.set_hexpand(True)
            hbox.pack_start(scale, True, True, 0)
            box.add(hbox)
            scales[key] = scale

        dialog.show_all()
        if dialog.run() == Gtk.ResponseType.OK:
            input_gain = int(scales["input_gain_percent"].get_value())
            output_gain = int(scales["output_gain_percent"].get_value())
            speaker_gain = int(scales["speaker_gain_percent"].get_value())
            headphone_gain = int(scales["headphone_gain_percent"].get_value())
            self.config["input_gain_percent"] = input_gain
            self.config["output_gain_percent"] = output_gain
            self.config["speaker_gain_percent"] = speaker_gain
            self.config["headphone_gain_percent"] = headphone_gain
            save_config(self.config)

            def _apply_gains():
                input_ok = self.pipeline.set_input_gain(input_gain)
                output_ok = self.pipeline.set_output_gain(output_gain)
                route_ok = self.pipeline.set_route_gains(speaker_gain, headphone_gain)
                if not input_ok or not output_ok or not route_ok:
                    GLib.idle_add(
                        self._show_error,
                        "Could not apply one or more audio gains. "
                        "Saved values will be retried when ClearVoice starts.",
                    )

            threading.Thread(target=_apply_gains, daemon=True).start()
        dialog.destroy()

    def _on_nc(self, item):
        self.config["noise_cancellation"]["enabled"] = item.get_active()
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart()

    def _on_studio_voice(self, item):
        self.config["studio_voice"]["enabled"] = item.get_active()
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
        self._quitting = True
        self._route_restart_pending = False
        if self._pw_monitor:
            self._pw_monitor.stop()
        self.pipeline.stop()
        save_config(self.config)
        Gtk.main_quit()

    def _on_popup(self, icon, button, timestamp):
        self.menu.popup(
            None, None, Gtk.StatusIcon.position_menu, icon, button, timestamp
        )

    # ── Helpers ──

    def _async_restart(self, route_restart: bool = False):
        """Restart the pipeline off the GTK thread."""
        if route_restart:
            if self._route_restart_pending:
                return
            self._route_restart_pending = True

        def _do():
            if route_restart and (self._quitting or not self.config.get("enabled", True)):
                GLib.idle_add(self._finish_async_restart, None, None, True)
                return
            ok, msg = self.pipeline.restart()
            GLib.idle_add(self._finish_async_restart, ok, msg, route_restart)

        threading.Thread(target=_do, daemon=True).start()

    def _finish_async_restart(self, ok: bool | None, msg: str | None, route_restart: bool):
        if route_restart:
            self._route_restart_pending = False
        if ok is None:
            return False
        self._update_icon()
        self._update_status()
        if not ok:
            self._show_error(msg)
            self._mi_enable.set_active(False)
        return False

    def _on_route_tick(self):
        if (
            self._quitting
            or not self.config.get("enabled", True)
            or not self.pipeline.running
            or self.pipeline.transitioning
            or self._route_probe_pending
            or self._route_restart_pending
        ):
            return True
        self._route_probe_pending = True

        def _probe():
            try:
                mode = self.pipeline.detect_headphone_mode()
            except Exception as exc:
                log.warning("Could not probe output route: %s", exc)
                mode = None
            GLib.idle_add(self._on_route_probe_complete, mode)

        threading.Thread(target=_probe, daemon=True).start()
        return True

    def _on_route_probe_complete(self, mode: bool | None):
        self._route_probe_pending = False
        if (
            mode is not None
            and not self._quitting
            and self.config.get("enabled", True)
            and self.pipeline.running
            and not self.pipeline.transitioning
            and mode != self.pipeline.headphone_mode
        ):
            self._async_restart(route_restart=True)
        return False

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

    def _update_status(self, nodes_active: bool = False):
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
            if self.pipeline.headphone_mode:
                parts.append("HP")
            tag = "+".join(parts) or "enabled"
            state = "Processing" if nodes_active else "Standby"
            self._mi_status.set_label(f"{state} [{tag}]")
        else:
            self._mi_status.set_label("Off")

    def _on_pw_state_change(self, nodes_active: bool):
        """Called by PipeWireMonitor when node state changes (GTK thread)."""
        self._update_icon(nodes_active=nodes_active)
        self._update_status(nodes_active=nodes_active)

    def _on_health_tick(self):
        """Process liveness check only — state is event-driven."""
        if self.pipeline.running:
            if not self.pipeline.check_health():
                log.warning("Health check failed — restarting pipeline")
                self._async_restart()
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


def _acquire_instance_lock() -> bool:
    """Ensure only one ClearVoice instance runs. Returns True if we got the lock."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    # Check for stale PID
    if PIDFILE.exists():
        try:
            old_pid = int(PIDFILE.read_text().strip())
            os.kill(old_pid, 0)  # check if alive
            # Process exists — another instance is running
            return False
        except (ProcessLookupError, ValueError):
            pass  # stale pidfile, we can take over
        except PermissionError:
            return False  # alive but we can't signal it
    PIDFILE.write_text(str(os.getpid()))
    atexit.register(lambda: PIDFILE.unlink(missing_ok=True))
    return True


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    if not _acquire_instance_lock():
        print(f"{APP_NAME} is already running.", file=sys.stderr)
        sys.exit(0)

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
        tray._quitting = True
        tray._route_restart_pending = False
        if tray._pw_monitor:
            tray._pw_monitor.stop()
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
