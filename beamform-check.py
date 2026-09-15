#!/usr/bin/env python3
"""Interactive no-save validation for a two-microphone array."""

import argparse
import json
import math
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


RATE = 48_000
CHANNELS = 2
WARMUP_SECONDS = 2
MAX_LAG_SAMPLES = 12
PHASES = [
    ("silence", 3, "Stay silent and keep the laptop still."),
    ("center", 5, "Speak normally from the center of the screen."),
    ("near_left", 4, "Gently rub near the LEFT microphone pinhole."),
    ("near_right", 4, "Gently rub near the RIGHT microphone pinhole."),
    ("speech_left", 5, "Move your head left of the laptop and speak normally."),
    ("speech_right", 5, "Move your head right of the laptop and speak normally."),
]
CONFIG_FILE = Path.home() / ".config" / "clearvoice" / "config.json"
RUNTIME_DIR = Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
) / "clearvoice"
RESULTS_FILE = RUNTIME_DIR / "beamforming-check.json"
MANAGER_ENV = {**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"}


def db(value: float) -> float:
    return 20 * math.log10(max(value, 1e-12))


def load_config() -> dict:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def pw_objects() -> list[dict]:
    result = subprocess.run(
        ["pw-dump", "-r", "pipewire-0-manager"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return json.loads(result.stdout)


def find_base_source(config: dict, objects: list[dict]) -> str:
    sources = {
        obj.get("info", {}).get("props", {}).get("node.name", "")
        for obj in objects
        if obj.get("type") == "PipeWire:Interface:Node"
        and obj.get("info", {}).get("props", {}).get("media.class")
        == "Audio/Source"
    }
    for candidate in (
        config.get("source_device"),
        config.get("previous_default_source"),
    ):
        if candidate in sources and not candidate.startswith("clearvoice"):
            return candidate
    physical = [
        name
        for name in sources
        if name and not name.startswith("clearvoice") and ".monitor" not in name
    ]
    if len(physical) != 1:
        raise RuntimeError("Select the laptop microphone in ClearVoice first")
    return physical[0]


def ensure_laptop_mode(config: dict):
    sink_name = config.get("previous_default_sink")
    if not sink_name:
        return
    result = subprocess.run(
        ["pactl", "--format=json", "list", "sinks"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    sink = next(
        (item for item in json.loads(result.stdout) if item.get("name") == sink_name),
        None,
    )
    if sink and sink.get("active_port") == "analog-output-headphones":
        raise RuntimeError("Unplug the headset so the laptop microphone array is active")


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def capture_pass(source: str) -> np.ndarray:
    total_seconds = WARMUP_SECONDS + sum(duration for _, duration, _ in PHASES)
    expected_bytes = total_seconds * RATE * CHANNELS * 4
    process = subprocess.Popen(
        [
            "pw-record",
            f"--target={source}",
            f"--rate={RATE}",
            f"--channels={CHANNELS}",
            "--format=f32",
            "--raw",
            "--properties=stream.dont-remix=true",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=MANAGER_ENV,
    )
    payload = []

    def reader():
        payload.append(read_exact(process.stdout, expected_bytes))

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        print(f"\nWarming up capture for {WARMUP_SECONDS}s…", flush=True)
        time.sleep(WARMUP_SECONDS)
        for _, duration, instruction in PHASES:
            print(f"\n{instruction}", flush=True)
            for remaining in range(duration, 0, -1):
                print(f"  {remaining:2d}s remaining", end="\r", flush=True)
                time.sleep(1)
            print(" " * 24, end="\r", flush=True)
    finally:
        thread.join(timeout=3)
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        thread.join(timeout=1)

    data = payload[0] if payload else b""
    if len(data) != expected_bytes:
        error = process.stderr.read().decode(errors="replace").strip()
        raise RuntimeError(
            f"Capture returned {len(data)} of {expected_bytes} bytes: {error}"
        )
    return np.frombuffer(data, dtype=np.float32).reshape(-1, CHANNELS)


def best_lag(left: np.ndarray, right: np.ndarray) -> tuple[int, float]:
    left = left - np.mean(left)
    right = right - np.mean(right)
    best = (0, -1.0)
    for lag in range(-MAX_LAG_SAMPLES, MAX_LAG_SAMPLES + 1):
        if lag < 0:
            a, b = left[:lag], right[-lag:]
        elif lag > 0:
            a, b = left[lag:], right[:-lag]
        else:
            a, b = left, right
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        correlation = float(np.dot(a, b) / denominator) if denominator else 0.0
        if correlation > best[1]:
            best = (lag, correlation)
    return best


def analyze(samples: np.ndarray) -> dict:
    result = {}
    cursor = WARMUP_SECONDS * RATE
    trim = RATE // 4
    for name, duration, _ in PHASES:
        block = np.nan_to_num(samples[cursor + trim : cursor + duration * RATE - trim])
        cursor += duration * RATE
        left, right = block[:, 0], block[:, 1]
        left_rms = float(np.sqrt(np.mean(left * left)))
        right_rms = float(np.sqrt(np.mean(right * right)))
        lag, correlation = best_lag(left, right)
        result[name] = {
            "left_rms_dbfs": db(left_rms),
            "right_rms_dbfs": db(right_rms),
            "left_minus_right_db": db(left_rms) - db(right_rms),
            "correlation": correlation,
            "left_minus_right_lag_samples": lag,
            "lag_microseconds": lag * 1_000_000 / RATE,
            "maximum_channel_difference": float(np.max(np.abs(left - right))),
        }

    left_near = result["near_left"]["left_minus_right_db"]
    right_near = result["near_right"]["left_minus_right_db"]
    dominance_swaps = (
        left_near * right_near < 0
        and abs(left_near) >= 6
        and abs(right_near) >= 6
    )
    side_lags = (
        result["speech_left"]["left_minus_right_lag_samples"],
        result["speech_right"]["left_minus_right_lag_samples"],
    )
    lag_swaps = side_lags[0] * side_lags[1] < 0
    channel_order = "FL=left" if left_near > right_near else "FR=left"
    result["verdict"] = {
        "independent_channels": dominance_swaps,
        "side_delay_direction_swaps": lag_swaps,
        "channel_order": channel_order if dominance_swaps else "unknown",
        "suitable_for_beamforming": dominance_swaps,
    }
    return result


def print_report(report: dict):
    for phase, _, _ in PHASES:
        values = report[phase]
        print(
            f"  {phase:12} L {values['left_rms_dbfs']:6.1f} dBFS, "
            f"R {values['right_rms_dbfs']:6.1f} dBFS, "
            f"L−R {values['left_minus_right_db']:+5.1f} dB, "
            f"corr {values['correlation']:+.3f}, "
            f"lag {values['left_minus_right_lag_samples']:+d} samples"
        )
    verdict = report["verdict"]
    print("\n  Independent channels:", "PASS" if verdict["independent_channels"] else "INCONCLUSIVE")
    print("  Side-delay reversal:", "PASS" if verdict["side_delay_direction_swaps"] else "INCONCLUSIVE")
    print("  Channel order:", verdict["channel_order"])
    print("  Beamforming candidate:", "YES" if verdict["suitable_for_beamforming"] else "NOT YET")


def save_results(source: str, passes: list[dict]):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_retained": False,
        "sample_rate": RATE,
        "channels": CHANNELS,
        "source": source,
        "passes": passes,
    }
    temporary = RESULTS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(RESULTS_FILE)


def self_test():
    total_seconds = WARMUP_SECONDS + sum(duration for _, duration, _ in PHASES)
    rng = np.random.default_rng(7)
    samples = np.zeros((total_seconds * RATE, CHANNELS), dtype=np.float32)
    cursor = WARMUP_SECONDS * RATE
    for name, duration, _ in PHASES:
        size = duration * RATE
        signal = rng.normal(0, 0.05, size).astype(np.float32)
        if name == "near_left":
            samples[cursor : cursor + size, 0] = signal
            samples[cursor : cursor + size, 1] = signal * 0.1
        elif name == "near_right":
            samples[cursor : cursor + size, 0] = signal * 0.1
            samples[cursor : cursor + size, 1] = signal
        elif name == "speech_left":
            samples[cursor : cursor + size, 0] = signal
            samples[cursor : cursor + size, 1] = np.roll(signal, 4)
        elif name == "speech_right":
            samples[cursor : cursor + size, 0] = np.roll(signal, 4)
            samples[cursor : cursor + size, 1] = signal
        else:
            samples[cursor : cursor + size, 0] = signal
            samples[cursor : cursor + size, 1] = signal
        cursor += size

    report = analyze(samples)
    assert report["verdict"]["independent_channels"]
    assert report["verdict"]["side_delay_direction_swaps"]
    assert report["verdict"]["suitable_for_beamforming"]
    print("Beamforming spatial self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    config = load_config()
    ensure_laptop_mode(config)
    source = find_base_source(config, pw_objects())
    passes = []

    print("ClearVoice two-microphone spatial test")
    print(f"Raw source: {source}")
    print("Audio stays in memory; only aggregate spatial metrics are saved.")
    input("\nPress Enter to begin pass 1…")

    while True:
        number = len(passes) + 1
        print(f"\n=== PASS {number} ===", flush=True)
        report = analyze(capture_pass(source))
        report["pass"] = number
        passes.append(report)
        save_results(source, passes)
        print_report(report)
        print(f"\nMetrics saved to {RESULTS_FILE}")
        if input("Another pass? [y/N] ").strip().lower() not in {"y", "yes"}:
            break

    print(f"\nDone. Tell me when to inspect {RESULTS_FILE}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpatial test cancelled; no audio was saved.")
