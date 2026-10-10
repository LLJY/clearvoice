#!/usr/bin/env python3
"""ClearVoice — PipeWire noise cancellation, beamforming & AEC system tray tool.

Creates a virtual microphone with DeepFilterNet noise cancellation,
WebRTC-based beamforming, and acoustic echo cancellation via PipeWire.
"""

import atexit
import json
import logging
import math
import os
import re
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

# Private plugins: a user build (build-*.sh) overrides the system package.
USER_LIB_DIR = Path.home() / ".local/lib/clearvoice"
SYSTEM_LIB_DIR = Path("/usr/lib/clearvoice")


def _lib_path(relative: str) -> Path:
    """Return the user-built file if present, else the system-packaged one."""
    user = USER_LIB_DIR / relative
    return user if user.is_file() else SYSTEM_LIB_DIR / relative


PRIVATE_AEC_PLUGIN = _lib_path("spa-0.2/aec/libspa-aec-webrtc.so")
PRIVATE_SPA_ROOT = PRIVATE_AEC_PLUGIN.parents[1]
SYSTEM_SPA_ROOT = Path("/usr/lib/spa-0.2")
REQUIRED_PIPEWIRE_SERIES = "1.6"
DEEPFILTER_RT_PRIORITY = 10

# PipeWire node names
VIRTUAL_MIC_NAME = "clearvoice_source"
VIRTUAL_MIC_DESC = "ClearVoice"
EC_SOURCE_NAME = "clearvoice_beamformed"
EC_SOURCE_DESC = "ClearVoice Beamformed"

# LADSPA plugin
DEEPFILTER_SO = "libdeep_filter_ladspa.so"
DEEPFILTER_LABEL_MONO = "deep_filter_mono"
DEEPFILTER_LABEL_STEREO = "deep_filter_stereo"
CLEARVOICE_LADSPA_PLUGIN = _lib_path("ladspa/libclearvoice_ladspa.so")
NOISE_MODELS = {
    "stock": "Stock DeepFilterNet",
    "dfn3-ll": "DeepFilterNet3-LL (constant latency)",
    "dfn3-ll-int8": "DeepFilterNet3-LL int8",
    "fastenhancer-b": "FastEnhancer-B",
    "fastenhancer-s": "FastEnhancer-S",
    "fastenhancer-m": "FastEnhancer-M",
}
CLEARVOICE_LADSPA_LABELS = {
    "dfn3-ll": "clearvoice_dfn3_ll_mono",
    "dfn3-ll-int8": "clearvoice_dfn3_ll_int8_mono",
    "fastenhancer-b": "clearvoice_fastenhancer_b_mono",
    "fastenhancer-s": "clearvoice_fastenhancer_s_mono",
    "fastenhancer-m": "clearvoice_fastenhancer_m_mono",
}

LADSPA_SEARCH_PATHS = [
    "/usr/lib/ladspa",
    "/usr/lib64/ladspa",
    "/usr/local/lib/ladspa",
    str(Path.home() / ".ladspa"),
]

# Speaker enhancement config (ships with the project)
SPEAKER_CHAIN_CONF = Path(__file__).parent / "speaker-chain.conf"
SPEAKER_SINK_NAME = "clearvoice_speakers"
ORPHAN_PROCESS_PATTERN = r"^pipewire -c .*/clearvoice/"
# Coalesce bursts of graph events (Bluetooth connects emit several) into one route probe.
ROUTE_EVENT_DEBOUNCE_MS = 150

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
    "lock_output_volume": True,
    "noise_cancellation": {
        "enabled": True,
        "model": "stock",
        "latency_ms": 35,
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
    nc = config["noise_cancellation"]
    if (
        not isinstance(nc.get("model"), str)
        or nc.get("model") not in NOISE_MODELS
    ):
        log.warning(
            "Unknown noise_cancellation.model %r; using stock", nc.get("model")
        )
        nc["model"] = "stock"
    nc["latency_ms"] = _clamp_latency_ms(nc.get("latency_ms", 35))
    return config


def _clamp_latency_ms(value) -> int:
    try:
        return max(10, min(200, int(value)))
    except (TypeError, ValueError, OverflowError):
        return 35


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
    for cmd in ("pipewire", "pw-dump", "pactl", "wpctl", "busctl"):
        if not shutil.which(cmd):
            missing.append(cmd)
    if not find_ladspa_plugin(DEEPFILTER_SO):
        missing.append(f"DeepFilterNet LADSPA ({DEEPFILTER_SO})")
    return missing


# ── PipeWire Helpers ──────────────────────────────────────────────────────────


def pw_dump_objects(manager: bool = False) -> list[dict] | None:
    """Return PipeWire objects, optionally through WirePlumber's manager remote."""
    try:
        result = subprocess.run(
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
        if result.returncode != 0:
            return None
        objects = json.loads(result.stdout)
        return objects if isinstance(objects, list) else None
    except Exception as exc:
        log.error("Failed to query PipeWire objects: %s", exc)
        return None


def pw_dump_matching(pattern: str, manager: bool = False) -> list[dict] | None:
    """Return objects whose name or path fnmatch-es *pattern* (pw-dump's own filter).

    About half the cost of a full dump. Callers still compare exact properties.
    """
    try:
        result = subprocess.run(
            ["pw-dump", pattern],
            capture_output=True,
            text=True,
            timeout=5,
            env=(
                {**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"}
                if manager
                else None
            ),
        )
        if result.returncode != 0:
            return None
        if not result.stdout.strip():
            return []  # pw-dump prints nothing when no object matches
        objects = json.loads(result.stdout)
        return objects if isinstance(objects, list) else None
    except Exception as exc:
        log.error("Failed to query PipeWire objects matching %s: %s", pattern, exc)
        return None


def pw_dump_named(name: str, manager: bool = False) -> list[dict] | None:
    """Return objects for one node name; names with glob characters get a full dump."""
    if any(char in name for char in "*?[("):
        return pw_dump_objects(manager)
    return pw_dump_matching(name, manager)


def _node_names(objects: list[dict]) -> set[str]:
    return {
        str(((obj.get("info") or {}).get("props") or {}).get("node.name"))
        for obj in objects
        if isinstance(obj, dict) and obj.get("type") == "PipeWire:Interface:Node"
    }


def pw_wait_for_nodes_gone(prefix: str, timeout: float = 1.0) -> bool:
    """Wait until no node named *prefix*... exists (exited children may linger briefly)."""
    deadline = time.monotonic() + timeout
    while True:
        objects = pw_dump_matching(f"{prefix}*")
        if objects is not None and not any(
            name.startswith(prefix) for name in _node_names(objects)
        ):
            return True
        if time.monotonic() >= deadline:
            log.warning("PipeWire nodes %s* still present after %.1f s", prefix, timeout)
            return False
        time.sleep(0.02)


def _pw_output_ports(objects: list[dict], node_id: int) -> list[dict]:
    """Return output port properties for one PipeWire node."""
    ports = []
    for obj in objects:
        if obj.get("type") != "PipeWire:Interface:Port":
            continue
        info = obj.get("info", {})
        props = info.get("props", {})
        direction = props.get("port.direction", info.get("direction"))
        if str(props.get("node.id")) == str(node_id) and direction in ("out", "output"):
            ports.append(props)
    return ports


def pw_source_has_separate_fl_fr(source_name: str) -> bool:
    """Return whether a manager-visible source exposes individual FL and FR ports."""
    objects = pw_dump_objects(manager=True)
    if objects is None:
        return False
    nodes = [
        obj
        for obj in objects
        if obj.get("type") == "PipeWire:Interface:Node"
        and obj.get("info", {}).get("props", {}).get("node.name") == source_name
    ]
    if len(nodes) != 1:
        return False
    channels = {
        props.get("audio.channel") for props in _pw_output_ports(objects, nodes[0]["id"])
    }
    return "FL" in channels and "FR" in channels


def pw_pipewire_versions() -> tuple[str | None, str | None]:
    """Return PipeWire's compiled and linked libpipewire versions."""
    try:
        result = subprocess.run(
            ["pipewire", "--version"], capture_output=True, text=True, timeout=3
        )
        if result.returncode != 0:
            return None, None
        compiled = re.search(
            r"^Compiled with libpipewire\s+(\S+)$", result.stdout, re.MULTILINE
        )
        linked = re.search(
            r"^Linked with libpipewire\s+(\S+)$", result.stdout, re.MULTILINE
        )
        return (
            compiled.group(1) if compiled else None,
            linked.group(1) if linked else None,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None


def serialize_mic_geometry(geometry: str) -> str:
    """Validate two microphone points and serialize them for the WebRTC backend."""
    if not isinstance(geometry, str):
        raise ValueError("geometry must be comma-separated numeric coordinates")
    try:
        values = [float(value.strip()) for value in geometry.split(",")]
    except ValueError as exc:
        raise ValueError("geometry must contain numeric coordinates") from exc
    if len(values) != 6:
        raise ValueError("geometry must contain exactly two microphone points")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("geometry coordinates must be finite")
    if values[:3] == values[3:]:
        raise ValueError("microphone points must be distinct")
    return json.dumps([values[:3], values[3:]], separators=(",", ":"), allow_nan=False)


def _private_aec_plugin_path() -> Path | None:
    """Return the private AEC plugin's exact real path when it exists."""
    try:
        plugin = PRIVATE_AEC_PLUGIN.resolve(strict=True)
    except OSError:
        return None
    return plugin if plugin.is_file() else None


def pw_private_aec_plugin_loaded(pid: int, plugin: Path) -> bool:
    """Verify that *pid* mapped the expected private AEC plugin without deletion."""
    try:
        plugin = plugin.resolve(strict=True)
        inode = str(plugin.stat().st_ino)
        for line in Path(f"/proc/{pid}/maps").read_text().splitlines():
            if "(deleted)" in line:
                continue
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[4] == inode and fields[5] == str(plugin):
                return True
    except OSError:
        pass
    return False


def pw_wait_for_beamformed_source(
    node_name: str, pid: int, timeout: float = 6.0
) -> tuple[bool, str]:
    """Wait for one PID-owned, mono beamformer output on the manager remote."""
    deadline = time.monotonic() + timeout
    reason = "beamformer source did not appear"
    while time.monotonic() < deadline:
        objects = pw_dump_objects(manager=True)
        if objects is None:
            time.sleep(0.05)
            continue
        clients = [
            obj
            for obj in objects
            if obj.get("type") == "PipeWire:Interface:Client"
            and str(obj.get("info", {}).get("props", {}).get("application.process.id"))
            == str(pid)
        ]
        nodes = [
            obj
            for obj in objects
            if obj.get("type") == "PipeWire:Interface:Node"
            and obj.get("info", {}).get("props", {}).get("node.name") == node_name
            and obj.get("info", {}).get("props", {}).get("media.class") == "Audio/Source"
        ]
        if len(nodes) > 1:
            return False, "beamformer source name is ambiguous"
        if nodes:
            props = nodes[0].get("info", {}).get("props", {})
            if len(clients) != 1:
                reason = "beamformer process client did not appear"
                time.sleep(0.05)
                continue
            if str(props.get("client.id")) != str(clients[0]["id"]):
                return False, "beamformer source is not owned by its echo-cancel process"
            ports = _pw_output_ports(objects, nodes[0]["id"])
            if len(ports) == 1 and ports[0].get("audio.channel") == "MONO":
                return True, ""
            reason = "beamformer source did not expose exactly one MONO output port"
        time.sleep(0.05)
    return False, reason


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


def _pactl_list(kind: str) -> list[dict] | None:
    """Return PulseAudio-compatible *kind* (sinks, cards) data, or None on query errors."""
    try:
        result = subprocess.run(
            ["pactl", "--format=json", "list", kind],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode != 0:
            log.warning("Could not list %s: %s", kind, result.stderr.strip())
            return None
        entries = json.loads(result.stdout)
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict) for entry in entries
        ):
            log.warning("Unexpected %s list response", kind)
            return None
        return entries
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log.warning("Could not list %s: %s", kind, exc)
        return None


def pactl_list_sinks() -> list[dict] | None:
    return _pactl_list("sinks")


def pactl_list_cards() -> list[dict] | None:
    return _pactl_list("cards")


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


# Class-of-device form factors that play into the room even when they have a mic.
BLUETOOTH_LOUDSPEAKER_FORM_FACTORS = {"speaker", "portable", "hifi", "car"}


def _bluetooth_output_card(sink: dict) -> str | None:
    """Card name of a Bluetooth output that is not a loudspeaker, else None."""
    properties = sink.get("properties")
    if not isinstance(properties, dict) or properties.get("device.api") != "bluez5":
        return None
    if properties.get("device.form_factor") in BLUETOOTH_LOUDSPEAKER_FORM_FACTORS:
        return None
    card = properties.get("device.name")
    return card if isinstance(card, str) else None


def _card_has_mic(card: dict) -> bool:
    """A mic shows up as an available card profile with an input (HSP/HFP head unit)."""
    profiles = card.get("profiles")
    return isinstance(profiles, dict) and any(
        isinstance(profile, dict)
        and profile.get("sources", 0) > 0
        and profile.get("available", True)
        for profile in profiles.values()
    )


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
    for obj in pw_dump_named(node_name, manager) or []:
        props = (obj.get("info") or {}).get("props") or {}
        if (
            obj.get("type") == "PipeWire:Interface:Node"
            and props.get("node.name") == node_name
            and props.get("media.class") in ("Audio/Sink", "Audio/Source")
        ):
            return obj["id"]
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
    """Monitor whether an external client consumes the ClearVoice source.

    Fires *on_state_change(bool)* when a link to ``clearvoice_source``
    appears or disappears. Zero CPU when nothing changes.
    """

    def __init__(self, on_state_change: callable, on_link_removed=None, on_route_change=None):
        self.on_state_change = on_state_change
        self.on_link_removed = on_link_removed
        self.on_route_change = on_route_change
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._active = False
        self._enabled = False
        self._objects: dict[int, dict] = {}
        self._route_signature: frozenset | None = None

    @property
    def nodes_active(self) -> bool:
        return self._active

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._enabled = True
        self._proc = subprocess.Popen(
            [
                "pw-dump",
                "-r",
                "pipewire-0-manager",
                "--monitor",
                "--no-colors",
            ],
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
        """Update the graph snapshot and detect external source consumers."""
        try:
            objects = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if not isinstance(objects, list):
            return

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("id")
            if not isinstance(object_id, int):
                continue
            if obj.get("type") is None or obj.get("info") is None:
                previous = self._objects.get(object_id)
                if (
                    self._active
                    and self.on_link_removed
                    and previous
                    and previous.get("type") == "PipeWire:Interface:Link"
                ):
                    self.on_link_removed()
                self._objects.pop(object_id, None)
            else:
                self._objects[object_id] = obj

        nodes = {
            obj["id"]: obj.get("info", {}).get("props", {})
            for obj in self._objects.values()
            if obj.get("type") == "PipeWire:Interface:Node"
        }
        final_nodes = {
            node_id
            for node_id, props in nodes.items()
            if props.get("node.name") == VIRTUAL_MIC_NAME
        }
        ports = {
            obj["id"]: obj.get("info", {}).get("props", {})
            for obj in self._objects.values()
            if obj.get("type") == "PipeWire:Interface:Port"
        }
        final_outputs = {
            port_id
            for port_id, props in ports.items()
            if props.get("node.id") in final_nodes
            and props.get("port.direction") == "out"
        }
        now_active = False
        for obj in self._objects.values():
            if obj.get("type") != "PipeWire:Interface:Link":
                continue
            info = obj.get("info", {})
            if info.get("output-port-id") not in final_outputs:
                continue
            input_props = ports.get(info.get("input-port-id"), {})
            target_props = nodes.get(input_props.get("node.id"), {})
            if not str(target_props.get("node.name", "")).startswith("clearvoice"):
                now_active = True
                break

        if now_active != self._active:
            self._active = now_active
            GLib.idle_add(self.on_state_change, now_active)

        route_signature = _route_signature(self._objects.values())
        if (
            self.on_route_change
            and self._route_signature is not None
            and route_signature != self._route_signature
        ):
            GLib.idle_add(self.on_route_change)
        self._route_signature = route_signature


def _route_signature(objects) -> frozenset:
    """What decides the output route: physical sinks plus device routes and profiles.

    Volumes live in Route props and are left out, so volume changes do not count.
    """
    items = set()
    for obj in objects:
        info = obj.get("info") or {}
        props = info.get("props") or {}
        if obj.get("type") == "PipeWire:Interface:Device":
            params = info.get("params") or {}

            def entries(key):
                return tuple(sorted(
                    (str(entry.get("name")), str(entry.get("available")))
                    for entry in params.get(key) or []
                    if isinstance(entry, dict)
                ))

            items.add(("device", props.get("device.name"), entries("Route"), entries("EnumProfile")))
        elif obj.get("type") == "PipeWire:Interface:Node" and props.get("media.class") == "Audio/Sink":
            name = props.get("node.name")
            if isinstance(name, str) and not name.startswith("clearvoice"):
                items.add(("sink", name))
    return frozenset(items)


def pw_link_ports(output_port: str, input_port: str, connect: bool) -> bool:
    """Create or remove one managed PipeWire link."""
    command = ["pw-link"]
    if not connect:
        command.append("--disconnect")
    command.extend((output_port, input_port))
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            env={**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"},
        )
        if result.returncode != 0:
            log.warning(
                "Could not %s link %s to %s: %s",
                "connect" if connect else "disconnect",
                output_port,
                input_port,
                result.stderr.strip(),
            )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("Could not update PipeWire link: %s", exc)
        return False


def _pw_link_present(
    objects: list[dict], output_endpoint: str, input_endpoint: str
) -> bool | None:
    """Check exact node/port names in a successful snapshot."""
    nodes = {}
    for obj in objects:
        if isinstance(obj, dict) and obj.get("type") == "PipeWire:Interface:Node":
            props = (obj.get("info") or {}).get("props") or {}
            nodes.setdefault(props.get("node.name"), []).append(str(obj.get("id")))
    ports = {}
    for obj in objects:
        if not isinstance(obj, dict) or obj.get("type") != "PipeWire:Interface:Port":
            continue
        info = obj.get("info") or {}
        props = info.get("props") or {}
        node_id = str(props.get("node.id"))
        port_name = props.get("port.name")
        direction = props.get("port.direction", info.get("direction"))
        if port_name:
            ports.setdefault((node_id, port_name, direction), []).append(
                str(obj.get("id"))
            )

    def resolve(endpoint: str, direction: tuple[str, ...]) -> str | None:
        try:
            node_name, port_name = endpoint.rsplit(":", 1)
        except ValueError:
            return None
        node_ids = nodes.get(node_name, ())
        if len(node_ids) != 1:
            return None
        matches = [
            port_id
            for port_direction in direction
            for port_id in ports.get((node_ids[0], port_name, port_direction), ())
        ]
        return matches[0] if len(matches) == 1 else None

    output_id = resolve(output_endpoint, ("out", "output"))
    input_id = resolve(input_endpoint, ("in", "input"))
    if output_id is None or input_id is None:
        return None
    return any(
        obj.get("type") == "PipeWire:Interface:Link"
        and str((obj.get("info") or {}).get("output-port-id")) == output_id
        and str((obj.get("info") or {}).get("input-port-id")) == input_id
        for obj in objects
        if isinstance(obj, dict)
    )


def _deepfilter_thread(
    pid: int, timeout: float = 2.0, worker_name: str | None = None
) -> tuple[int | None, str]:
    """Identify the one non-graph PipeWire worker from the plugin's known layout."""
    try:
        result = subprocess.run(
            ["ps", "-L", "-p", str(pid), "-o", "tid=,comm=,cls="],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode != 0:
        return None, result.stderr.strip() or "could not inspect process threads"

    candidates = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        tid, comm, scheduling_class = fields
        try:
            thread_id = int(tid)
        except ValueError:
            continue
        if thread_id == pid or comm == "module-rt" or comm.startswith("data-loop"):
            continue
        candidates.append((thread_id, comm, scheduling_class))
    if not candidates:
        return None, "inference worker has not appeared"
    if worker_name is not None:
        matches = [
            candidate for candidate in candidates if candidate[1] == worker_name
        ]
        if not matches:
            return None, f"{worker_name} worker has not appeared"
        if len(matches) != 1 or matches[0][2] not in ("TS", "RR"):
            return None, "ambiguous PipeWire thread layout: " + ", ".join(
                f"{tid}:{comm}/{scheduling_class}"
                for tid, comm, scheduling_class in matches
            )
        return matches[0][0], ""
    if (
        len(candidates) != 1
        or candidates[0][1] != "pipewire"
        or candidates[0][2] not in ("TS", "RR")
    ):
        return None, "ambiguous PipeWire thread layout: " + ", ".join(
            f"{tid}:{comm}/{scheduling_class}"
            for tid, comm, scheduling_class in candidates
        )
    return candidates[0][0], ""


def _deepfilter_worker_runtime_ns(pid: int, worker: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/task/{worker}/schedstat").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def promote_deepfilter_worker(
    pid: int,
    timeout: float = 10.0,
    should_cancel=None,
    worker_name: str | None = None,
) -> int | None:
    """Promote the caught-up DeepFilter worker below PipeWire's graph priority."""
    deadline = time.monotonic() + timeout
    reason = "inference worker has not appeared"
    worker_busy = False
    while time.monotonic() < deadline:
        if should_cancel and should_cancel():
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        worker, reason = _deepfilter_thread(
            pid, timeout=min(2, remaining), worker_name=worker_name
        )
        if worker is not None:
            try:
                # RTKit sets SCHED_RESET_ON_FORK alongside the policy; ignore that flag.
                policy = os.sched_getscheduler(worker) & ~os.SCHED_RESET_ON_FORK
                priority = os.sched_getparam(worker).sched_priority
            except OSError:
                time.sleep(0.01)
                continue
            if should_cancel and should_cancel():
                return None
            if policy == os.SCHED_RR:
                if priority == DEEPFILTER_RT_PRIORITY:
                    log.debug(
                        "DeepFilter worker %d already uses SCHED_RR %d",
                        worker,
                        priority,
                    )
                    return worker
                log.warning(
                    "DeepFilter worker %d has unexpected SCHED_RR priority %d",
                    worker,
                    priority,
                )
                return None
            if policy != os.SCHED_OTHER:
                log.warning(
                    "DeepFilter worker %d has unexpected scheduling policy %d",
                    worker,
                    policy,
                )
                return None
            # Stock: backlog draining can exhaust PipeWire's RLIMIT_RTTIME before
            # promotion. cv-dsp-worker discards late frames and naps itself, so it skips
            # this gate (on battery DFN3-LL is busy > 50% and would never qualify).
            if worker_name is None:
                reason = "worker did not become idle before promotion timeout"
                runtime_before = _deepfilter_worker_runtime_ns(pid, worker)
                if runtime_before is None:
                    log.debug(
                        "Skipping DeepFilter promotion: schedstat unreadable for worker %d",
                        worker,
                    )
                    return None
                sample_started = time.monotonic()
                if deadline - sample_started < 0.25:
                    break
                time.sleep(0.25)
                if should_cancel and should_cancel():
                    return None
                runtime_after = _deepfilter_worker_runtime_ns(pid, worker)
                sample_ns = int((time.monotonic() - sample_started) * 1_000_000_000)
                if (
                    runtime_after is None
                    or sample_ns <= 0
                    or runtime_after < runtime_before
                ):
                    log.debug(
                        "Skipping DeepFilter promotion: invalid schedstat sample for worker %d",
                        worker,
                    )
                    return None
                if (runtime_after - runtime_before) * 2 >= sample_ns:
                    worker_busy = True
                    reason = "worker remained busy draining its startup backlog"
                    continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                result = subprocess.run(
                    [
                        "busctl",
                        "call",
                        "org.freedesktop.RealtimeKit1",
                        "/org/freedesktop/RealtimeKit1",
                        "org.freedesktop.RealtimeKit1",
                        "MakeThreadRealtimeWithPID",
                        "ttu",
                        str(pid),
                        str(worker),
                        str(DEEPFILTER_RT_PRIORITY),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=min(3, remaining / 2),
                )
                if result.returncode == 0:
                    if should_cancel and should_cancel():
                        return None
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        log.warning(
                            "DeepFilter worker promotion could not be verified before timeout"
                        )
                        return None
                    verified, reason = _deepfilter_thread(
                        pid, timeout=min(2, remaining), worker_name=worker_name
                    )
                    try:
                        verified_policy = (
                            os.sched_getscheduler(verified) & ~os.SCHED_RESET_ON_FORK
                        )
                        verified_priority = os.sched_getparam(verified).sched_priority
                    except (OSError, TypeError):
                        verified_policy = verified_priority = None
                    if (
                        verified == worker
                        and verified_policy == os.SCHED_RR
                        and verified_priority == DEEPFILTER_RT_PRIORITY
                    ):
                        log.info(
                            "Promoted DeepFilter worker %d to SCHED_RR %d",
                            worker,
                            DEEPFILTER_RT_PRIORITY,
                        )
                        return worker
                    log.warning(
                        "DeepFilter worker promotion could not be verified: %s",
                        reason
                        or f"tid={verified}, policy={verified_policy}, "
                        f"priority={verified_priority}",
                    )
                    return None
                log.warning("RTKit rejected DeepFilter worker: %s", result.stderr.strip())
                return None
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("Could not promote DeepFilter worker: %s", exc)
                return None
        elif reason.startswith("ambiguous"):
            log.warning("Skipping DeepFilter promotion: %s", reason)
            return None
        time.sleep(0.01)
    if not should_cancel or not should_cancel():
        if worker_busy:
            log.warning("Skipping DeepFilter promotion: %s", reason)
        else:
            log.warning(
                "DeepFilter worker thread did not appear for RTKit promotion: %s",
                reason,
            )
    return None


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
    return node_name in _node_names(pw_dump_named(node_name) or [])


def pw_wait_for_node(node_name: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pw_node_exists(node_name):
            return True
        time.sleep(0.05)
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
    model: str = "stock",
    latency_ms: int = 35,
) -> str:
    """Build a PipeWire config that loads a DeepFilterNet filter-chain."""
    if not isinstance(model, str) or model not in NOISE_MODELS:
        model = "stock"
    latency_ms = _clamp_latency_ms(latency_ms)
    label = CLEARVOICE_LADSPA_LABELS.get(model, DEEPFILTER_LABEL_MONO)
    if model in ("stock", "dfn3-ll", "dfn3-ll-int8"):
        extra_latency_control = (
            f'                            "Latency (ms)" = {latency_ms}\n'
            if model != "stock"
            else ""
        )
        plugin_controls = (
            "                        control = {\n"
            f'                            "Attenuation Limit (dB)" = {attenuation_db}\n'
            f'                            "Min processing threshold (dB)" = {min_proc_db}\n'
            f'                            "Max ERB processing threshold (dB)" = {max_erb_db}\n'
            f'                            "Max DF processing threshold (dB)" = {max_df_db}\n'
            f'                            "Post Filter Beta" = {post_filter_beta}\n'
            f"{extra_latency_control}"
            "                        }\n"
        )
    else:
        plugin_controls = (
            "                        control = {\n"
            f'                            "Latency (ms)" = {latency_ms}\n'
            "                        }\n"
        )
    requested_latency_props = (
        "                node.latency = 256/48000\n" if model != "stock" else ""
    )
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
        "    cpu.zero.denormals = true\n"
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
        "    # ponytail: diagnostic 150ms soft limit sends SIGXCPU before unchanged 200ms hard SIGKILL.\n"
        "    { name = libpipewire-module-rt\n"
        "        args = { nice.level = -11 rt.time.soft = 150000 rt.time.hard = 200000 }\n"
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
        f"                        label  = {label}\n"
        f"{plugin_controls}"
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
        f"{requested_latency_props}"
        "            }\n"
        "            playback.props = {\n"
        f'                node.name        = "{VIRTUAL_MIC_NAME}"\n'
        f'                node.description = "{VIRTUAL_MIC_DESC}"\n'
        "                media.class      = Audio/Source\n"
        "                audio.rate       = 48000\n"
        f"{requested_latency_props}"
        "                session.suspend-timeout-seconds = 0\n"
        "                state.restore-props = false\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "]\n"
    )


def _speaker_chain_conf(physical_sink: str) -> str:
    """The shipped speaker-chain template, playing into this machine's physical sink."""
    template = SPEAKER_CHAIN_CONF.read_text()
    conf, count = re.subn(
        r'^(\s*target\.object\s*=\s*)"[^"]*"',
        lambda match: f'{match.group(1)}"{physical_sink}"',
        template,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise OSError(f"{SPEAKER_CHAIN_CONF}: expected one target.object, found {count}")
    return conf


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

    aec_parts = [
        "webrtc.noise_suppression=false",
        "webrtc.high_pass_filter=false",
        "webrtc.gain_control=false",
        "webrtc.voice_detection=false",
        "webrtc.transient_suppression=false",
    ]
    capture_props = ""
    if beamforming:
        mic_geometry = serialize_mic_geometry(mic_geometry)
        aec_parts.extend(
            [
                "webrtc.beamforming=true",
                f"webrtc.mic-geometry={mic_geometry}",
                "webrtc.target-direction=[1.5707963,0,1]",
            ]
        )
        capture_props = (
            "                audio.rate = 48000\n"
            "                audio.channels = 2\n"
            "                audio.position = [ FL FR ]\n"
            "                stream.dont-remix = true\n"
        )
    aec_args = " ".join(aec_parts)
    restore_props = "state.restore-props = false" if not is_intermediate else ""

    return (
        "# ClearVoice echo-cancel (auto-generated)\n"
        "context.properties = {\n"
        "    log.level = 0\n"
        "    cpu.zero.denormals = true\n"
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
        "    # ponytail: diagnostic 150ms soft limit sends SIGXCPU before unchanged 200ms hard SIGKILL.\n"
        "    { name = libpipewire-module-rt\n"
        "        args = { nice.level = -11 rt.time.soft = 150000 rt.time.hard = 200000 }\n"
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
        "                node.autoconnect = false\n"
        f"                {target_line}\n"
        f"{capture_props}"
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
        "                node.autoconnect = false\n"
        "            }\n"
        "            playback.props = {\n"
        '                node.name    = "clearvoice_ec_playback"\n'
        "                node.passive = true\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "]\n"
    )


def _parse_clearvoice_stats(line: str) -> dict | None:
    match = re.match(r"clearvoice-ladspa stats label=(\S+) (.+)$", line.strip())
    if not match:
        return None
    values = dict(re.findall(r"(\w+)=(\d+)", match.group(2)))
    if not {"processed", "concealed"} <= values.keys():
        return None
    return {
        "label": match.group(1),
        "processed": int(values["processed"]),
        "concealed": int(values["concealed"]),
    }


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
        model = self.config.get("noise_cancellation", {}).get("model", "stock")
        if not isinstance(model, str) or model not in NOISE_MODELS:
            log.warning("Unknown noise_cancellation.model %r; using stock", model)
            self.config.setdefault("noise_cancellation", {})["model"] = "stock"
        self._fc_proc: subprocess.Popen | None = None
        self._ec_proc: subprocess.Popen | None = None
        self._spk_proc: subprocess.Popen | None = None
        self._running = False
        self._transitioning = False  # True during stop/start — suppresses health checks
        self._lock = threading.Lock()
        self._demand_lock = threading.Lock()
        self._base_mic_node: str | None = None
        self._physical_sink: str | None = None
        self._playback_sink: str | None = None
        self._headphone_mode = False
        self._bluetooth_headset: str | None = None
        self._route_confirmed = False
        self._mic_demand = False
        self._demand_revision = 0
        self._reconcile_requested = False
        self._reconcile_running = False
        self._reconcile_thread: threading.Thread | None = None
        self._generation = 0
        self._shutdown_requested = False
        self._plugin_fallback = False
        self._fallback_notice: str | None = None
        self._fc_model: str | None = None
        self._fc_log_reader = None  # same inode as the filter-chain's stderr
        self._fc_log_pending = ""
        self._fc_stats_previous: tuple[int, int] | None = None
        self._fc_stats_warning_at: float | None = None
        self._fc_crash_times: list[float] = []
        self._fc_crash_counted = False

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
    def bluetooth_headset(self) -> str | None:
        return self._bluetooth_headset

    @property
    def _speakers_bypassed(self) -> bool:
        """Headphones or a Bluetooth headset: no speaker echo, so only NC runs."""
        return self._headphone_mode or self._bluetooth_headset is not None

    @property
    def nc_enabled(self) -> bool:
        return self.config["noise_cancellation"]["enabled"]

    @property
    def bf_enabled(self) -> bool:
        return self.config["beamforming"]["enabled"] and not self._speakers_bypassed

    @property
    def aec_enabled(self) -> bool:
        return self.config["echo_cancellation"]["enabled"] and not self._speakers_bypassed

    @property
    def ec_needed(self) -> bool:
        return self.bf_enabled or self.aec_enabled

    @property
    def spk_enabled(self) -> bool:
        return (
            self.config.get("speaker_enhancement", {}).get("enabled", False)
            and not self._speakers_bypassed
        )

    @property
    def studio_enabled(self) -> bool:
        return self.config.get("studio_voice", {}).get("enabled", True)

    @property
    def noise_model(self) -> str:
        selected = self.config.get("noise_cancellation", {}).get("model", "stock")
        if isinstance(selected, str) and selected in NOISE_MODELS:
            return selected
        return "stock"

    @property
    def active_noise_model(self) -> str:
        return self._fc_model or ("stock" if self._plugin_fallback else self.noise_model)

    @property
    def plugin_fallback_active(self) -> bool:
        return self._plugin_fallback

    def _activate_stock_fallback(self, reason: str):
        if self._plugin_fallback:
            return
        self._plugin_fallback = True
        message = (
            f"{NOISE_MODELS.get(self.noise_model, 'Selected noise model')} failed; "
            f"using stock DeepFilterNet for this session: {reason}"
        )
        self._fallback_notice = message
        log.error(message)

    def take_fallback_notice(self) -> str | None:
        message, self._fallback_notice = self._fallback_notice, None
        return message

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

    def detect_bluetooth_headset(self) -> str | None:
        """Return the sink of a connected Bluetooth headset (output with a mic).

        Mic-less Bluetooth outputs and loudspeakers are ignored. Keeps the current
        headset if PipeWire cannot be queried.
        """
        sinks = pactl_list_sinks()
        if sinks is None:
            return self._bluetooth_headset
        outputs = [
            (sink.get("name"), card)
            for sink in sinks
            if isinstance(sink.get("name"), str) and (card := _bluetooth_output_card(sink))
        ]
        if not outputs:
            return None
        cards = pactl_list_cards()
        if cards is None:
            return self._bluetooth_headset
        with_mic = {c.get("name") for c in cards if _card_has_mic(c)}
        names = [name for name, card in outputs if card in with_mic]
        if self._bluetooth_headset in names:
            return self._bluetooth_headset
        return names[0] if names else None

    def route_changed(self, headphone_mode: bool | None, bluetooth_headset: str | None) -> bool:
        """Whether a probed route needs a restart; a Bluetooth headset takes precedence."""
        if bluetooth_headset != self._bluetooth_headset:
            return True
        return (
            bluetooth_headset is None
            and headphone_mode is not None
            and headphone_mode != self._headphone_mode
        )

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

    @staticmethod
    def _publish_output_volume_lock(locked: bool) -> bool:
        return wp_set_setting(
            "clearvoice.lock-output-volume", str(bool(locked)).lower()
        )

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

    def set_output_volume_lock(self, locked: bool) -> bool:
        self.config["lock_output_volume"] = bool(locked)
        if not self._running:
            return True
        policy_set = self._publish_output_volume_lock(locked)
        return policy_set and (not locked or self.set_output_gain(100))

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

    def _set_playback_sink(self, sink_name: str, gain_percent: int | None) -> bool:
        """Set the sink volume (unless None) and default, then move non-ClearVoice streams."""
        gain_set = gain_percent is None or pw_set_node_volume(sink_name, gain_percent)
        sink_id = pw_find_node_id(sink_name)
        if sink_id is None:
            log.warning("Could not find playback sink %s", sink_name)
            return False
        default_set = pw_set_default_sink(sink_id)
        if not default_set:
            log.warning("Could not set default playback sink to %s", sink_name)
            return False
        self._playback_sink = sink_name
        if self._physical_sink:
            pw_move_playback_streams(self._physical_sink, sink_name)
        return gain_set

    def _set_physical_playback(self, sink_name: str, gain_percent: int):
        self._set_playback_sink(sink_name, gain_percent)
        if self._playback_sink != sink_name:
            log.error("Could not select physical playback sink %s; AEC reference unavailable", sink_name)

    def _stop_failed_speaker_chain(self):
        proc = self._spk_proc
        self._spk_proc = None
        if proc is None or proc.poll() is not None:
            return
        pid = getattr(proc, "pid", "unknown")
        log.warning("Stopping failed speaker-chain process (pid %s)", pid)
        try:
            proc.terminate()
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            log.warning(
                "Failed speaker-chain SIGTERM timed out (pid %s); escalating to SIGKILL",
                pid,
            )
            try:
                proc.kill()
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                log.error("Failed speaker-chain process (pid %s) survived SIGKILL", pid)
        except OSError as exc:
            log.warning("Could not stop failed speaker-chain process (pid %s): %s", pid, exc)

    def _start_playback_output(self):
        """Activate the Bluetooth headset, headphone or speaker playback route."""
        self._playback_sink = None
        if self._bluetooth_headset:
            # Bluetooth volume is the device's own; only select the sink.
            self._set_playback_sink(self._bluetooth_headset, None)
            if self._playback_sink == self._bluetooth_headset:
                log.info("Bluetooth headset %s: only noise cancellation runs", self._bluetooth_headset)
            else:
                log.warning("Could not select Bluetooth headset sink %s", self._bluetooth_headset)
            return
        physical_sink = self._physical_sink
        if not physical_sink:
            log.warning("No physical playback sink found; leaving playback route unchanged")
            return
        if self._headphone_mode:
            self._set_physical_playback(
                physical_sink, self.config["headphone_gain_percent"]
            )
            return
        if not self.spk_enabled or not SPEAKER_CHAIN_CONF.is_file():
            if not self.spk_enabled:
                self._set_physical_playback(
                    physical_sink, self.config["speaker_gain_percent"]
                )
            else:
                log.warning("Speaker chain config not found; using physical speakers")
                self._set_physical_playback(
                    physical_sink, self.config["speaker_gain_percent"]
                )
            return

        pw_set_node_volume(physical_sink, 100)
        log.info("Spawning speaker-chain process")
        try:
            conf_path = RUNTIME_DIR / "speaker-chain.conf"
            conf_path.write_text(_speaker_chain_conf(physical_sink))
            with open(RUNTIME_DIR / "speaker-chain.log", "a") as spk_log:
                spk_log.write(
                    f"\n--- speaker-chain start {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                )
                spk_log.flush()
                self._spk_proc = self._spawn_child(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=spk_log,
                )
        except OSError as exc:
            log.warning("Could not spawn speaker-chain; using physical speakers: %s", exc)
            self._set_physical_playback(physical_sink, self.config["speaker_gain_percent"])
            return
        if self._spk_proc is None:
            return
        if self._shutdown_pending():
            self._stop_failed_speaker_chain()
            return
        if pw_wait_for_node(SPEAKER_SINK_NAME, timeout=6.0):
            self._set_playback_sink(
                SPEAKER_SINK_NAME, self.config["speaker_gain_percent"]
            )
            if self._playback_sink == SPEAKER_SINK_NAME:
                log.info("Speaker chain ready: %s", SPEAKER_SINK_NAME)
                return
            log.warning("Could not select speaker-chain sink; falling back to physical speakers")
        else:
            log.warning("Speaker chain failed to start; using physical speakers")
        self._stop_failed_speaker_chain()
        self._set_physical_playback(physical_sink, self.config["speaker_gain_percent"])

    @staticmethod
    def _echo_child_env(beamforming: bool) -> dict[str, str]:
        """Select the SPA tree deterministically without changing module lookup."""
        env = os.environ.copy()
        env["SPA_PLUGIN_DIR"] = (
            f"{PRIVATE_SPA_ROOT}:{SYSTEM_SPA_ROOT}" if beamforming else str(SYSTEM_SPA_ROOT)
        )
        return env

    def set_lock_base_mic_audio(self, locked: bool) -> bool:
        self.config["lock_base_mic_audio"] = bool(locked)
        if self._running:
            return self._publish_lock_state(self.config["lock_base_mic_audio"])
        return True

    def set_mic_demand(self, active: bool) -> bool:
        """Publish the latest mic intent and queue off-GTK graph reconciliation."""
        active = bool(active)
        with self._demand_lock:
            if self._shutdown_requested:
                return False
            if active != self._mic_demand:
                self._demand_revision += 1
                self._mic_demand = active
                log.info("Mic DSP demand %s", "active" if active else "standby")
        self._request_mic_reconcile()
        return True

    def request_shutdown(self):
        """Cancel pending startup/reconciliation before asynchronous cleanup."""
        with self._demand_lock:
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            self._reconcile_requested = False
            if self._mic_demand:
                self._demand_revision += 1
                self._mic_demand = False

    def _shutdown_pending(self) -> bool:
        with self._demand_lock:
            return self._shutdown_requested

    def _spawn_child(self, command, **kwargs):
        with self._demand_lock:
            if self._shutdown_requested:
                return None
            return subprocess.Popen(command, **kwargs)

    def _request_mic_reconcile(self):
        with self._demand_lock:
            if self._shutdown_requested:
                return
            self._reconcile_requested = True
            if self._reconcile_running:
                return
            self._reconcile_running = True
            self._reconcile_thread = threading.Thread(
                target=self._mic_reconcile_loop,
                name="clearvoice-mic-reconcile",
                daemon=True,
            )
            thread = self._reconcile_thread
        try:
            thread.start()
        except RuntimeError:
            with self._demand_lock:
                self._reconcile_running = False
            log.exception("Could not start mic demand reconciler")

    def _mic_reconcile_loop(self):
        while True:
            with self._demand_lock:
                if self._shutdown_requested or not self._reconcile_requested:
                    self._reconcile_running = False
                    return
                self._reconcile_requested = False
                active = self._mic_demand
                revision = self._demand_revision
            try:
                generation = None
                with self._lock:
                    if (
                        self._running
                        and not self._transitioning
                        and not self._shutdown_requested
                    ):
                        generation = self._generation
                        self._reconcile_aec_links(active, revision, generation)
                # The promotion wait (up to 10 s) runs outside the lifecycle lock so health
                # ticks keep running; it cancels on generation/process/demand changes.
                if generation is not None:
                    self._reconcile_deepfilter(active, revision, generation)
            except Exception:
                log.exception("Mic demand reconciliation failed")
            with self._demand_lock:
                if revision != self._demand_revision:
                    self._reconcile_requested = True
                if self._reconcile_requested:
                    continue
                self._reconcile_running = False
                return

    def _demand_is_current(
        self, active: bool, revision: int, generation: int, proc=None
    ) -> bool:
        with self._demand_lock:
            if (
                self._shutdown_requested
                or self._mic_demand != active
                or self._demand_revision != revision
            ):
                return False
        return (
            self._generation == generation
            and self._running
            and not self._transitioning
            and (proc is None or (self._fc_proc is proc and proc.poll() is None))
        )

    def _aec_link_pairs(self) -> list[tuple[str, str]]:
        if not self.ec_needed or not self._base_mic_node:
            return []
        source = self._base_mic_node
        pairs = [
            (f"{source}:capture_FL", "clearvoice_ec_capture:input_FL"),
            (f"{source}:capture_FR", "clearvoice_ec_capture:input_FR"),
        ]
        if self.aec_enabled:
            reference = self._playback_sink
            if reference:
                pairs.extend(
                    [
                        (f"{reference}:monitor_FL", "clearvoice_ec_sink:input_FL"),
                        (f"{reference}:monitor_FR", "clearvoice_ec_sink:input_FR"),
                    ]
                )
            else:
                log.warning("AEC reference unavailable: no playback sink was selected")
        return pairs

    def _filter_capture_pairs(self, objects: list[dict]) -> list[tuple[str, str]]:
        if not self.nc_enabled or not self._base_mic_node:
            return []
        source = EC_SOURCE_NAME if self.ec_needed else self._base_mic_node
        target = "clearvoice_capture"

        def channels(node_name: str, direction: str, prefix: str):
            nodes = [
                str(obj.get("id"))
                for obj in objects
                if isinstance(obj, dict)
                and obj.get("type") == "PipeWire:Interface:Node"
                and ((obj.get("info") or {}).get("props") or {}).get("node.name")
                == node_name
            ]
            if len(nodes) != 1:
                return None
            result = {}
            for obj in objects:
                if not isinstance(obj, dict) or obj.get("type") != "PipeWire:Interface:Port":
                    continue
                info = obj.get("info") or {}
                props = info.get("props") or {}
                name = props.get("port.name")
                port_direction = props.get("port.direction", info.get("direction"))
                if (
                    str(props.get("node.id")) == nodes[0]
                    and port_direction
                    in (direction, "output" if direction == "out" else "input")
                    and isinstance(name, str)
                    and name.startswith(prefix)
                ):
                    channel = name.removeprefix(prefix)
                    if channel not in ("MONO", "FL", "FR") or channel in result:
                        return None
                    result[channel] = name
            return result or None

        outputs = channels(source, "out", "capture_")
        inputs = channels(target, "in", "input_")
        if outputs is None or inputs is None:
            log.warning(
                "Could not identify unambiguous %s -> %s filter ports", source, target
            )
            return []
        output_channels, input_channels = set(outputs), set(inputs)
        if output_channels == input_channels == {"MONO"}:
            channels_to_link = (("MONO", "MONO"),)
        elif (
            input_channels == {"MONO"}
            and "FL" in output_channels
            and "MONO" not in output_channels
        ):
            channels_to_link = (("FL", "MONO"),)
        elif output_channels == input_channels == {"FL", "FR"}:
            channels_to_link = (("FL", "FL"), ("FR", "FR"))
        else:
            log.warning(
                "Ambiguous capture channels for %s -> %s (outputs=%s inputs=%s)",
                source,
                target,
                sorted(output_channels),
                sorted(input_channels),
            )
            return []
        return [
            (
                f"{source}:{outputs[output_channel]}",
                f"{target}:{inputs[input_channel]}",
            )
            for output_channel, input_channel in channels_to_link
        ]

    def _reconcile_aec_links(
        self, active: bool, revision: int, generation: int
    ):
        if not self._base_mic_node or (not self.ec_needed and not (active and self.nc_enabled)):
            return
        objects = pw_dump_objects(manager=True)
        if objects is None:
            log.warning("Could not inspect PipeWire links; retrying on next health check")
            return
        pairs = self._aec_link_pairs()
        if active:
            pairs.extend(self._filter_capture_pairs(objects))
        missing = []
        for pair in pairs:
            present = _pw_link_present(objects, *pair)
            if present is None:
                log.warning(
                    "Could not unambiguously inspect PipeWire link %s -> %s", *pair
                )
            elif active and not present:
                missing.append(pair)
            elif not active and present:
                if not self._demand_is_current(False, revision, generation):
                    return
                pw_link_ports(*pair, connect=False)

        if not active:
            return

        added = []

        def rollback_added():
            if self._generation == generation:
                for output_port, input_port in added:
                    pw_link_ports(output_port, input_port, False)

        for output_port, input_port in missing:
            if not self._demand_is_current(True, revision, generation):
                rollback_added()
                return
            if not pw_link_ports(output_port, input_port, True):
                rollback_added()
                return
            added.append((output_port, input_port))
            if not self._demand_is_current(True, revision, generation):
                rollback_added()
                return

    def _reconcile_deepfilter(
        self, active: bool, revision: int, generation: int
    ):
        if not self._demand_is_current(active, revision, generation):
            return
        if not active or not self.nc_enabled:
            return
        proc = self._fc_proc
        if not proc or proc.poll() is not None:
            return

        def should_cancel():
            return not self._demand_is_current(True, revision, generation, proc)

        if self._fc_model in CLEARVOICE_LADSPA_LABELS:
            promote_deepfilter_worker(
                proc.pid, should_cancel=should_cancel, worker_name="cv-dsp-worker"
            )
        else:
            promote_deepfilter_worker(proc.pid, should_cancel=should_cancel)

    # ── Start / Stop ──

    def start(self) -> tuple[bool, str]:
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"
        with self._lock:
            if self._shutdown_pending():
                return False, "Pipeline shutdown requested"
            try:
                result = self._start_locked()
            except Exception as exc:
                log.exception("Pipeline start failed")
                result = self._fail_start(str(exc))
        if result[0]:
            self._request_mic_reconcile()
        return result

    @staticmethod
    def _cleanup_orphans():
        """Kill any orphaned ClearVoice PipeWire processes from a previous crash."""
        try:
            result = subprocess.run(
                # Anchored: only our own `pipewire -c` children, never shells/editors
                # whose command line merely mentions these paths.
                ["pgrep", "-u", str(os.getuid()), "-f", ORPHAN_PROCESS_PATTERN],
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

    def _wait_for_filter_chain_node(
        self, proc: subprocess.Popen, timeout: float = 6.0
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False
            if pw_node_exists(VIRTUAL_MIC_NAME):
                return proc.poll() is None
            time.sleep(0.05)
        return False

    def _stop_filter_chain_attempt(self):
        proc = self._fc_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        self._fc_proc = None
        self._fc_model = None
        self._close_filter_chain_log()

    def _close_filter_chain_log(self):
        reader, self._fc_log_reader = self._fc_log_reader, None
        if reader is not None:
            reader.close()

    def _launch_filter_chain(
        self, model: str, plugin_path: str, target_source: str, nc: dict
    ) -> tuple[bool, str]:
        conf = _pw_conf_filter_chain(
            plugin_path=plugin_path,
            attenuation_db=nc.get("attenuation_limit_db", 100),
            min_proc_db=nc.get("min_processing_threshold_db", -15),
            max_erb_db=nc.get("max_erb_threshold_db", 35),
            max_df_db=nc.get("max_df_threshold_db", 35),
            post_filter_beta=nc.get("post_filter_beta", 0.0),
            target_source=target_source,
            studio_voice=self.studio_enabled,
            model=model,
            latency_ms=nc.get("latency_ms", 35),
        )
        conf_path = RUNTIME_DIR / "filter-chain.conf"
        conf_path.write_text(conf)
        self._fc_model = model
        self._fc_crash_counted = False
        self._fc_log_pending = ""
        self._fc_stats_previous = None

        log.info("Spawning filter-chain process (%s)", model)
        try:
            with open(RUNTIME_DIR / "filter-chain.log", "ab") as fc_log:
                fc_log.write(
                    (
                        f"\n--- filter-chain start "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                    ).encode()
                )
                fc_log.flush()
                self._close_filter_chain_log()
                # Reopen the child's stderr inode (not the path) so only this attempt's
                # new bytes are read, even if the path is later replaced.
                self._fc_log_reader = open(f"/proc/self/fd/{fc_log.fileno()}", "rb")
                self._fc_log_reader.seek(0, os.SEEK_END)
                self._fc_proc = self._spawn_child(
                    ["pipewire", "-c", str(conf_path)],
                    stdout=subprocess.DEVNULL,
                    stderr=fc_log,
                )
        except OSError as exc:
            self._fc_proc = None
            self._fc_model = None
            return False, str(exc)
        if self._fc_proc is None:
            self._fc_model = None
            return False, "Pipeline shutdown requested"
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"

        if not self._wait_for_filter_chain_node(self._fc_proc):
            stderr = self._read_recent_log("filter-chain")
            self._stop_filter_chain_attempt()
            return False, f"Filter-chain failed to start: {stderr}"

        log.info("Filter-chain ready: %s (%s)", VIRTUAL_MIC_NAME, model)
        return True, ""

    def _start_filter_chain(
        self, target_source: str, stock_plugin_path: str
    ) -> tuple[bool, str]:
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"
        nc = self.config["noise_cancellation"]
        model = "stock" if self._plugin_fallback else self.noise_model
        plugin_path = stock_plugin_path
        if model != "stock":
            if not CLEARVOICE_LADSPA_PLUGIN.is_file():
                self._activate_stock_fallback(
                    f"LADSPA plugin not found: {CLEARVOICE_LADSPA_PLUGIN}"
                )
                model = "stock"
            else:
                plugin_path = str(CLEARVOICE_LADSPA_PLUGIN)

        ok, reason = self._launch_filter_chain(model, plugin_path, target_source, nc)
        if ok or model == "stock" or self._shutdown_pending():
            if self._shutdown_pending():
                return False, "Pipeline shutdown requested"
            return ok, reason

        self._activate_stock_fallback(reason)
        if self._shutdown_pending():
            return False, reason
        return self._launch_filter_chain(
            "stock", stock_plugin_path, target_source, nc
        )

    def _start_locked(self) -> tuple[bool, str]:
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"
        if self._running:
            return True, "Already running"

        self._generation += 1
        self._running = False
        self._base_mic_node = None
        self._playback_sink = None
        self._physical_sink = self._resolve_physical_sink()
        self._refresh_headphone_mode()
        self._bluetooth_headset = self.detect_bluetooth_headset()

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
        bf_geometry = ""
        private_aec_plugin = None

        if needs_mic:
            stock_plugin_path = find_ladspa_plugin(DEEPFILTER_SO)
            if self.nc_enabled and not stock_plugin_path:
                return False, f"LADSPA plugin not found: {DEEPFILTER_SO}"

            source = self._resolve_source()
            if not source:
                return False, "No audio source device found"

        if self.bf_enabled:
            private_aec_plugin = _private_aec_plugin_path()
            if private_aec_plugin is None:
                return self._fail_start(
                    f"Private beamformer plugin not found: {PRIVATE_AEC_PLUGIN}"
                )
            compiled, linked = pw_pipewire_versions()
            # Point releases keep the SPA AEC interface; rebuild per major.minor series.
            series = {".".join((v or "").split(".")[:2]) for v in (compiled, linked)}
            if series != {REQUIRED_PIPEWIRE_SERIES}:
                return self._fail_start(
                    "Beamforming requires PipeWire compiled and linked with "
                    f"{REQUIRED_PIPEWIRE_SERIES}.x (got {compiled or 'unknown'}/"
                    f"{linked or 'unknown'})"
                )
            try:
                custom = self.config["beamforming"].get("custom_geometry")
                geometry = custom or MIC_PRESETS.get(
                    self.config["beamforming"].get(
                        "preset", "laptop-dual-50mm"
                    ),
                    {},
                ).get("geometry", "")
                serialize_mic_geometry(geometry)
                bf_geometry = geometry
            except ValueError as exc:
                return self._fail_start(f"Invalid beamforming geometry: {exc}")
            if not pw_source_has_separate_fl_fr(source):
                return self._fail_start(
                    "Beamforming requires separate FL and FR output ports on the selected source"
                )

        log.info(
            "Starting pipeline — source=%s mic=%s spk=%s hp=%s bt=%s",
            source,
            needs_mic,
            self.spk_enabled,
            self._headphone_mode,
            self._bluetooth_headset,
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
        if self._shutdown_pending():
            return self._fail_start("Pipeline shutdown requested")

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
        final_node = VIRTUAL_MIC_NAME

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

                conf = _pw_conf_echo_cancel(
                    target_source=source,
                    monitor_mode=self.aec_enabled,
                    beamforming=self.bf_enabled,
                    mic_geometry=bf_geometry,
                    source_name=ec_out_name,
                    source_desc=ec_out_desc,
                    is_intermediate=self.nc_enabled,
                )
                conf_path = RUNTIME_DIR / "echo-cancel.conf"
                conf_path.write_text(conf)

                log.info("Spawning echo-cancel process")
                with open(RUNTIME_DIR / "echo-cancel.log", "a") as ec_log:
                    ec_log.write(
                        "\n--- echo-cancel start "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} ---\n"
                    )
                    ec_log.flush()
                    self._ec_proc = self._spawn_child(
                        ["pipewire", "-c", str(conf_path)],
                        stdout=subprocess.DEVNULL,
                        stderr=ec_log,
                        env=self._echo_child_env(self.bf_enabled),
                    )
                if self._ec_proc is None:
                    return self._fail_start("Pipeline shutdown requested")
                if self._shutdown_pending():
                    return self._fail_start("Pipeline shutdown requested")

                if self.bf_enabled:
                    beamformer_ready, reason = pw_wait_for_beamformed_source(
                        ec_out_name, self._ec_proc.pid, timeout=6.0
                    )
                    if not beamformer_ready:
                        return self._fail_start(f"Beamformer verification failed: {reason}")
                    if not pw_private_aec_plugin_loaded(
                        self._ec_proc.pid, private_aec_plugin
                    ):
                        return self._fail_start(
                            "Beamformer verification failed: private AEC plugin is not mapped"
                        )
                elif not pw_wait_for_node(ec_out_name, timeout=6.0):
                    stderr = self._read_recent_log("echo-cancel")
                    return self._fail_start(f"Echo-cancel failed to start: {stderr}")

                log.info("Echo-cancel ready: %s", ec_out_name)
                final_node = ec_out_name

            # ── Stage 2: Filter-chain (DeepFilterNet) ──
            if self.nc_enabled:
                fc_target = EC_SOURCE_NAME if self.ec_needed else source
                if not stock_plugin_path:
                    return self._fail_start(
                        f"LADSPA plugin not found: {DEEPFILTER_SO}"
                    )
                started, reason = self._start_filter_chain(
                    fc_target, stock_plugin_path
                )
                if not started:
                    return self._fail_start(reason)
                final_node = VIRTUAL_MIC_NAME

            # ── Stage 3: Set mic as default + configured output gain ──
            if needs_mic:
                output_gain = (
                    100
                    if self.config["lock_output_volume"]
                    else self.config["output_gain_percent"]
                )
                output_gain_ready = self.set_output_gain(output_gain)
                if not pw_set_default_source(final_node):
                    return self._fail_start(
                        f"Could not set default source to {final_node}"
                    )
                if self.config["lock_base_mic_audio"] and not output_gain_ready:
                    return self._fail_start("Could not prepare ClearVoice output gain lock")
            else:
                self._restore_previous_default_source()

            lock_state = self.config["lock_base_mic_audio"] if needs_mic else False
            if not self._publish_lock_state(lock_state):
                return self._fail_start("Could not publish base mic audio lock policy")
            output_lock = self.config["lock_output_volume"] if needs_mic else False
            if not self._publish_output_volume_lock(output_lock):
                return self._fail_start("Could not publish ClearVoice volume lock")

            if self._shutdown_pending():
                return self._fail_start("Pipeline shutdown requested")
            self._running = True
            return True, "Pipeline active"

        except Exception as exc:
            log.exception("Pipeline start failed")
            return self._fail_start(str(exc))

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            return self._stop_locked()

    def _stop_locked(self, restore_defaults: bool = True) -> tuple[bool, str]:
        if not self._running:
            return True, "Already stopped"

        self._generation += 1
        was_transitioning = self._transitioning
        self._transitioning = True
        log.info("Stopping pipeline")

        self._publish_lock_state(False)
        self._publish_output_volume_lock(False)
        self._kill_all()
        if restore_defaults:
            self._restore_previous_defaults()
        self._base_mic_node = None
        self._playback_sink = None
        self._running = False
        if not was_transitioning:
            self._transitioning = False
        return True, "Pipeline stopped"

    def _fail_start(self, msg: str) -> tuple[bool, str]:
        self._running = False
        self._publish_lock_state(False)
        self._publish_output_volume_lock(False)
        self._kill_all()
        self._restore_previous_defaults()
        self._base_mic_node = None
        self._playback_sink = None
        return False, msg

    def _restore_previous_default_source(self):
        prev = self.config.get("previous_default_source")
        if prev:
            pw_set_default_source(prev)

    def _restore_previous_defaults(self):
        self._restore_previous_default_source()

        prev_sink = self.config.get("previous_default_sink")
        # Keep a connected Bluetooth headset selected instead of moving playback to the speakers.
        if prev_sink and not self.detect_bluetooth_headset():
            # Restore by name — find its node ID
            sink_id = pw_find_node_id(prev_sink)
            if sink_id:
                pw_set_default_sink(sink_id)

    def restart(self) -> tuple[bool, str]:
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"
        with self._lock:
            if self._shutdown_pending():
                return False, "Pipeline shutdown requested"
            self._transitioning = True
            try:
                # Keep ClearVoice as the configured default so apps return to the new
                # virtual mic as soon as it appears; a graph without one restores the
                # mic default in _start_locked, with the config that start actually reads.
                self._stop_locked(restore_defaults=False)
                pw_wait_for_nodes_gone("clearvoice_")
                result = (
                    (False, "Pipeline shutdown requested")
                    if self._shutdown_pending()
                    else self._start_locked()
                )
            except Exception as exc:
                log.exception("Pipeline restart failed")
                result = self._fail_start(str(exc))
            finally:
                self._transitioning = False
            if not result[0]:
                # Early start exits and shutdown skip _fail_start; restoring twice is harmless.
                self._restore_previous_defaults()
        if result[0]:
            self._request_mic_reconcile()
        return result

    def restart_filter_chain(self) -> tuple[bool, str]:
        """Relaunch only the noise filter; echo-cancel and the speaker chain stay up.

        Falls back to a full restart when the filter-chain is not part of the running graph.
        """
        if self._shutdown_pending():
            return False, "Pipeline shutdown requested"
        with self._lock:
            if self._shutdown_pending():
                return False, "Pipeline shutdown requested"
            scoped = (
                self._running
                and self.nc_enabled
                and self._base_mic_node is not None
                and self._fc_proc is not None
            )
            if scoped:
                self._transitioning = True
                self._generation += 1  # cancels reconcile/promotion of the old process
                try:
                    result = self._relaunch_filter_chain_locked()
                except Exception as exc:
                    log.exception("Filter-chain restart failed")
                    result = self._fail_start(str(exc))
                finally:
                    self._transitioning = False
        if not scoped:
            return self.restart()
        if result[0]:
            self._request_mic_reconcile()
        return result

    def _relaunch_filter_chain_locked(self) -> tuple[bool, str]:
        log.info("Restarting filter-chain only")
        self._stop_filter_chain_attempt()
        pw_wait_for_nodes_gone(VIRTUAL_MIC_NAME)
        pw_wait_for_nodes_gone("clearvoice_capture")
        stock_plugin_path = find_ladspa_plugin(DEEPFILTER_SO)
        if not stock_plugin_path:
            return self._fail_start(f"LADSPA plugin not found: {DEEPFILTER_SO}")
        target = EC_SOURCE_NAME if self.ec_needed else self._base_mic_node
        started, reason = self._start_filter_chain(target, stock_plugin_path)
        if not started:
            return self._fail_start(reason)
        output_gain_ready = self.set_output_gain(
            100 if self.config["lock_output_volume"] else self.config["output_gain_percent"]
        )
        if not pw_set_default_source(VIRTUAL_MIC_NAME):
            return self._fail_start(f"Could not set default source to {VIRTUAL_MIC_NAME}")
        if self.config["lock_base_mic_audio"] and not output_gain_ready:
            return self._fail_start("Could not prepare ClearVoice output gain lock")
        return True, "Filter-chain restarted"

    # ── Health ──

    def _consume_filter_chain_log(self) -> bool:
        reader = self._fc_log_reader
        if reader is None:
            return False
        try:
            # ponytail: truncation is only detected if the file shrank below our
            # position; ClearVoice never truncates this append-only log.
            if os.fstat(reader.fileno()).st_size < reader.tell():
                reader.seek(0)
                self._fc_log_pending = ""
                self._fc_stats_previous = None
            data = reader.read()
        except (OSError, ValueError):
            return False

        text = self._fc_log_pending + data.decode(errors="replace")
        lines = text.splitlines(keepends=True)
        self._fc_log_pending = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._fc_log_pending = lines.pop()

        fatal_seen = False
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("clearvoice-ladspa fatal "):
                self._activate_stock_fallback(line)
                fatal_seen = True
                continue
            stats = _parse_clearvoice_stats(line)
            if stats is None:
                continue
            current = (stats["processed"], stats["concealed"])
            previous = self._fc_stats_previous
            self._fc_stats_previous = current
            if previous is None:
                continue
            processed_delta = current[0] - previous[0]
            concealed_delta = current[1] - previous[1]
            if processed_delta <= 0 or concealed_delta < 0:
                continue
            ratio = concealed_delta / processed_delta
            now = time.monotonic()
            if ratio > 0.05 and (
                self._fc_stats_warning_at is None
                or now - self._fc_stats_warning_at >= 60
            ):
                self._fc_stats_warning_at = now
                log.warning(
                    "ClearVoice LADSPA concealment %.1f%% over stats window "
                    "(concealed=%d processed=%d); model=%s",
                    ratio * 100,
                    concealed_delta,
                    processed_delta,
                    stats["label"],
                )
        return fatal_seen

    def _record_filter_chain_crash(self, pid: int):
        if self._fc_crash_counted:
            return
        self._fc_crash_counted = True
        now = time.monotonic()
        self._fc_crash_times = [
            crashed_at
            for crashed_at in self._fc_crash_times
            if now - crashed_at <= 60
        ]
        self._fc_crash_times.append(now)
        if len(self._fc_crash_times) >= 2:
            self._activate_stock_fallback(
                f"filter-chain crashed twice within 60 seconds (latest pid={pid})"
            )

    def check_health(self) -> bool:
        # Never block the GTK thread; a busy lifecycle lock means a stop/start is in
        # progress, whose intentional terminations must not count as crashes.
        if not self._lock.acquire(blocking=False):
            return True
        try:
            return self._check_health_locked()
        finally:
            self._lock.release()

    def _check_health_locked(self) -> bool:
        if not self._running or self._transitioning:
            return True
        if (
            self._fc_model in CLEARVOICE_LADSPA_LABELS
            and self._consume_filter_chain_log()
        ):
            return False
        dead = []
        for attr, name in (
            ("_fc_proc", "filter-chain"),
            ("_ec_proc", "echo-cancel"),
            ("_spk_proc", "speaker-chain"),
        ):
            proc = getattr(self, attr)
            if proc:
                returncode = proc.poll()
                if returncode is not None:
                    dead.append((name, proc.pid, returncode))
        if dead:
            for name, pid, rc in dead:
                if (
                    name == "filter-chain"
                    and self._fc_model in CLEARVOICE_LADSPA_LABELS
                ):
                    self._record_filter_chain_crash(pid)
                if rc < 0:
                    try:
                        status = f"signal {signal.Signals(-rc).name} ({-rc})"
                    except ValueError:
                        status = f"signal {-rc}"
                else:
                    status = f"exit code {rc}"
                stderr = self._read_recent_log(name)
                log.error(
                    "%s died (pid=%d, %s); recent stderr: %r",
                    name,
                    pid,
                    status,
                    stderr,
                )
            return False
        return True

    # ── Internals ──

    @staticmethod
    def _read_recent_log(name: str, limit: int = 500) -> str:
        try:
            with open(RUNTIME_DIR / f"{name}.log", "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - limit))
                return stream.read().decode(errors="replace").strip() or "(empty)"
        except OSError:
            return "(no stderr captured)"

    def _kill_all(self):
        # Send SIGTERM to all processes first (non-blocking)
        procs = []
        for attr in ("_fc_proc", "_ec_proc", "_spk_proc"):
            proc: subprocess.Popen | None = getattr(self, attr)
            if proc is not None and proc.poll() is None:
                log.info(
                    "Sending intentional SIGTERM to %s (pid %s)",
                    attr,
                    getattr(proc, "pid", "unknown"),
                )
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
                log.warning(
                    "Intentional SIGTERM timed out for %s (pid %s); escalating to SIGKILL",
                    attr,
                    getattr(proc, "pid", "unknown"),
                )
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    log.warning(
                        "Process %s (pid %s) did not die after intentional SIGKILL",
                        attr,
                        getattr(proc, "pid", "unknown"),
                    )
            setattr(self, attr, None)
        self._fc_model = None
        self._close_filter_chain_log()


# ── Tray UI ───────────────────────────────────────────────────────────────────


class ClearVoiceTray:
    """System tray interface."""

    def __init__(self, pipeline: PipelineManager, config: dict):
        self.pipeline = pipeline
        self.config = config
        self._pw_monitor: PipeWireMonitor | None = None
        self._route_probe_pending = False
        self._route_restart_pending = False
        self._route_event_pending = False
        self._health_restart_pending = False
        self._quitting = False
        self._enable_converge_lock = threading.Lock()
        self._enable_converge_running = False
        self._enable_replay_mic = False
        self._enable_converge_thread: threading.Thread | None = None

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

        self._register_periodic_checks()

        # Jack-route probes run off the GTK thread.
        GLib.timeout_add_seconds(2, self._on_route_tick)

        # Event-driven node state monitor
        self._pw_monitor = PipeWireMonitor(
            on_state_change=self._on_pw_state_change,
            on_link_removed=self.pipeline._request_mic_reconcile,
            on_route_change=self._on_route_event,
        )
        self._pw_monitor.start()

        # Start pipeline if enabled in config (off GTK thread)
        if self.config.get("enabled", True):
            self._request_enabled_converge(replay_mic=True)

    def _register_periodic_checks(self):
        GLib.timeout_add_seconds(2, self._on_health_tick)
        GLib.timeout_add_seconds(10, self._on_reconcile_tick)

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

        self._mi_lock_output_volume = Gtk.CheckMenuItem(
            label="Lock ClearVoice Volume"
        )
        self._mi_lock_output_volume.set_active(self.config["lock_output_volume"])
        self._mi_lock_output_volume.connect("toggled", self._on_lock_output_volume)
        m.append(self._mi_lock_output_volume)

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

        mi_model = Gtk.MenuItem(label="    Noise Model")
        sub_model = Gtk.Menu()
        mi_model.set_submenu(sub_model)
        m.append(mi_model)
        model_group = []
        for model, label in NOISE_MODELS.items():
            ri = Gtk.RadioMenuItem(
                label=label, group=model_group[0] if model_group else None
            )
            ri.set_active(self.pipeline.noise_model == model)
            ri.connect("toggled", self._on_noise_model, model)
            sub_model.append(ri)
            model_group.append(ri)

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

        cur_preset = self.config["beamforming"].get("preset", "laptop-dual-50mm")
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
        if self._quitting:
            return
        enabled = item.get_active()
        with self._enable_converge_lock:
            self.config["enabled"] = enabled
        save_config(self.config)
        if not enabled:
            self._route_restart_pending = False
        self._request_enabled_converge()

    def _request_enabled_converge(self, replay_mic: bool = False):
        with self._enable_converge_lock:
            if self._quitting:
                return
            self._enable_replay_mic |= replay_mic
            if self._enable_converge_running:
                return
            self._enable_converge_running = True
            self._enable_converge_thread = threading.Thread(
                target=self._converge_enabled_state,
                name="clearvoice-enable-converge",
                daemon=True,
            )
            thread = self._enable_converge_thread
        try:
            thread.start()
        except RuntimeError:
            with self._enable_converge_lock:
                self._enable_converge_running = False
            log.exception("Could not start enable-state reconciler")

    def _converge_enabled_state(self):
        while True:
            with self._enable_converge_lock:
                if self._quitting:
                    self._enable_converge_running = False
                    return
                enabled = bool(self.config.get("enabled", True))

            if self.pipeline.running != enabled:
                try:
                    ok, msg = (
                        self.pipeline.start() if enabled else self.pipeline.stop()
                    )
                except Exception as exc:
                    log.exception("Could not converge ClearVoice enable state")
                    ok, msg = False, str(exc)
                if not ok:
                    with self._enable_converge_lock:
                        latest = bool(self.config.get("enabled", True))
                        if self._quitting:
                            self._enable_converge_running = False
                            return
                        if latest != enabled or self.pipeline.running == latest:
                            continue
                        self._enable_converge_running = False
                    GLib.idle_add(
                        self._finish_enabled_converge, enabled, False, msg
                    )
                    return
                continue

            with self._enable_converge_lock:
                if self._quitting:
                    self._enable_converge_running = False
                    return
                if enabled and self._enable_replay_mic:
                    self._enable_replay_mic = False
                    if self._pw_monitor:
                        self.pipeline.set_mic_demand(self._pw_monitor.nodes_active)
                latest = bool(self.config.get("enabled", True))
                if latest != self.pipeline.running or (
                    latest and self._enable_replay_mic
                ):
                    continue
                self._enable_converge_running = False
            GLib.idle_add(self._finish_enabled_converge, latest, True, "")
            return

    def _finish_enabled_converge(self, enabled: bool, ok: bool, msg: str):
        if self._quitting or bool(self.config.get("enabled", True)) != enabled:
            return False
        if ok and self.pipeline.running != enabled:
            return False
        self._update_icon()
        self._update_status()
        self._show_pending_fallback_notice()
        if not ok:
            log.error("Could not %s ClearVoice", "enable" if enabled else "disable")
            self._show_error(msg)
            if enabled:
                self._mi_enable.set_active(False)
        return False

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

    def _on_lock_output_volume(self, item):
        locked = item.get_active()
        previous = self.config["lock_output_volume"]
        if locked == previous:
            return
        self.config["lock_output_volume"] = locked
        if locked:
            self.config["output_gain_percent"] = 100
        save_config(self.config)
        item.set_sensitive(False)

        def _do():
            if self.pipeline.set_output_volume_lock(locked):
                GLib.idle_add(item.set_sensitive, True)
                return

            def _restore():
                self.config["lock_output_volume"] = previous
                save_config(self.config)
                item.set_active(previous)
                item.set_sensitive(True)
                self._show_error("Could not update ClearVoice volume lock")

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
            if key == "output_gain_percent" and self.config["lock_output_volume"]:
                scale.set_value(100)
                scale.set_sensitive(False)
            hbox.pack_start(scale, True, True, 0)
            box.add(hbox)
            scales[key] = scale

        dialog.show_all()
        if dialog.run() == Gtk.ResponseType.OK:
            input_gain = int(scales["input_gain_percent"].get_value())
            output_gain = (
                100
                if self.config["lock_output_volume"]
                else int(scales["output_gain_percent"].get_value())
            )
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
            self._async_restart(filter_only=True)

    def _on_atten(self, item, val):
        if not item.get_active():
            return
        self.config["noise_cancellation"]["attenuation_limit_db"] = val
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart(filter_only=True)

    def _on_noise_model(self, item, model):
        if not item.get_active():
            return
        self.config["noise_cancellation"]["model"] = model
        save_config(self.config)
        if self.pipeline.running:
            self._async_restart(filter_only=True)

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
                self._async_restart(filter_only=True)
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
                "  x1,y1,z1,x2,y2,z2\n\n"
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
                try:
                    serialize_mic_geometry(text)
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
        self._begin_shutdown()

    def _begin_shutdown(self):
        if self._quitting:
            return
        self._quitting = True
        self._route_restart_pending = False
        self.pipeline.request_shutdown()

        def _do():
            try:
                if self._pw_monitor:
                    self._pw_monitor.stop()
            except Exception:
                log.exception("Could not stop PipeWire monitor")
            try:
                self.pipeline.stop()
            except Exception:
                log.exception("Could not stop ClearVoice cleanly")
            try:
                save_config(self.config)
            except Exception:
                log.exception("Could not save ClearVoice configuration")
            GLib.idle_add(Gtk.main_quit)

        threading.Thread(target=_do, daemon=True).start()

    def _on_popup(self, icon, button, timestamp):
        self.menu.popup(
            None, None, Gtk.StatusIcon.position_menu, icon, button, timestamp
        )

    # ── Helpers ──

    def _async_restart(
        self, route_restart: bool = False, health_restart: bool = False, filter_only: bool = False
    ):
        """Restart the pipeline (or only the noise filter) off the GTK thread."""
        if self._quitting or not self.config.get("enabled", True):
            return
        if route_restart:
            if self._route_restart_pending:
                return
            self._route_restart_pending = True
        if health_restart:
            # One child death must not queue a restart per 2 s liveness tick.
            if self._health_restart_pending:
                return
            self._health_restart_pending = True

        def _do():
            try:
                if self._quitting or not self.config.get("enabled", True):
                    return
                ok, msg = (
                    self.pipeline.restart_filter_chain()
                    if filter_only
                    else self.pipeline.restart()
                )
                if self._quitting:
                    return
                if not self.config.get("enabled", True):
                    # Disabled mid-restart: let the single converger own the final state.
                    self._request_enabled_converge()
                    return
                GLib.idle_add(self._finish_async_restart, ok, msg, route_restart)
            finally:
                if health_restart:
                    self._health_restart_pending = False

        threading.Thread(target=_do, daemon=True).start()

    def _finish_async_restart(self, ok: bool | None, msg: str | None, route_restart: bool):
        if self._quitting:
            return False
        if route_restart:
            self._route_restart_pending = False
        if ok is None:
            return False
        self._update_icon()
        self._update_status()
        self._show_pending_fallback_notice()
        if not ok:
            self._show_error(msg)
            self._mi_enable.set_active(False)
        return False

    def _on_route_event(self):
        """A sink, jack route or device profile changed: probe soon, coalescing bursts."""
        if not self._quitting and not self._route_event_pending:
            self._route_event_pending = True
            GLib.timeout_add(ROUTE_EVENT_DEBOUNCE_MS, self._on_route_event_timeout)
        return False

    def _on_route_event_timeout(self):
        self._route_event_pending = False
        self._on_route_tick()
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
                route = (
                    self.pipeline.detect_headphone_mode(),
                    self.pipeline.detect_bluetooth_headset(),
                )
            except Exception as exc:
                log.warning("Could not probe output route: %s", exc)
                route = (None, self.pipeline.bluetooth_headset)
            GLib.idle_add(self._on_route_probe_complete, *route)

        threading.Thread(target=_probe, daemon=True).start()
        return True

    def _on_route_probe_complete(self, mode: bool | None, bluetooth_headset: str | None):
        self._route_probe_pending = False
        if (
            not self._quitting
            and self.config.get("enabled", True)
            and self.pipeline.running
            and not self.pipeline.transitioning
            and self.pipeline.route_changed(mode, bluetooth_headset)
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
                model = NOISE_MODELS.get(
                    self.pipeline.active_noise_model, NOISE_MODELS["stock"]
                )
                if self.pipeline.plugin_fallback_active:
                    model += " (fallback)"
                parts.append(f"NC:{model}")
            if self.pipeline.bf_enabled:
                parts.append("BF")
            if self.pipeline.aec_enabled:
                parts.append("AEC")
            if self.pipeline.spk_enabled:
                parts.append("SPK")
            if self.pipeline.headphone_mode and not self.pipeline.bluetooth_headset:
                parts.append("HP")
            if self.pipeline.bluetooth_headset:
                parts.append("BT")
            tag = "+".join(parts) or "enabled"
            state = "Processing" if nodes_active else "Standby"
            self._mi_status.set_label(f"{state} [{tag}]")
        else:
            self._mi_status.set_label("Off")

    def _show_pending_fallback_notice(self):
        message = self.pipeline.take_fallback_notice()
        if message:
            self._show_error(message)

    def _on_pw_state_change(self, nodes_active: bool):
        """Called when an external app starts or stops consuming the mic."""
        if self._quitting:
            return False
        if not self.pipeline.set_mic_demand(nodes_active):
            log.error("Could not update mic DSP demand links")
        self._update_icon(nodes_active=nodes_active)
        self._update_status(nodes_active=nodes_active)

    def _on_health_tick(self):
        """Process liveness check only — state is event-driven."""
        if self._quitting:
            return GLib.SOURCE_REMOVE
        if self.pipeline.running:
            if not self.pipeline.check_health():
                if not self._health_restart_pending:
                    log.warning("Health check failed — restarting pipeline")
                self._async_restart(health_restart=True)
        return True  # keep timer

    def _on_reconcile_tick(self):
        """Backstop event-driven link repair with a periodic graph check."""
        if self._quitting:
            return GLib.SOURCE_REMOVE
        if self.pipeline.running:
            self.pipeline._request_mic_reconcile()
        return True

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
        tray._begin_shutdown()
        return GLib.SOURCE_REMOVE

    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, _shutdown)
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, _shutdown)

    log.info("Tray ready — entering GTK main loop")
    Gtk.main()

    log.info("%s shutdown complete", APP_NAME)


if __name__ == "__main__":
    main()
