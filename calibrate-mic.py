#!/usr/bin/env python3
"""Interactive, no-audio-retention microphone calibration for ClearVoice."""

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
PHASES = [
    ("silence", 5, "Stay silent."),
    ("normal", 8, "Speak normally at your usual distance."),
    ("loud", 5, "Speak loudly, without moving closer."),
    ("sibilant", 4, 'Say: "six sleek swans swiftly swam south."'),
]
BANDS = [
    ("rumble", 20, 80),
    ("body", 80, 200),
    ("low_mid", 200, 500),
    ("mid", 500, 1_500),
    ("presence", 1_500, 4_000),
    ("sibilance", 4_000, 8_000),
    ("air", 8_000, 16_000),
]
CONFIG_FILE = Path.home() / ".config" / "clearvoice" / "config.json"
RUNTIME_DIR = Path(
    os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
) / "clearvoice"
RESULTS_FILE = RUNTIME_DIR / "calibration-results.json"
MANAGER_ENV = {**os.environ, "PIPEWIRE_REMOTE": "pipewire-0-manager"}


def db(value: float) -> float:
    return 20 * math.log10(max(value, 1e-12))


def list_sources() -> list[str]:
    result = subprocess.run(
        ["pw-dump", "-r", "pipewire-0-manager"],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return [
        obj.get("info", {}).get("props", {}).get("node.name", "")
        for obj in json.loads(result.stdout)
        if obj.get("type") == "PipeWire:Interface:Node"
        and obj.get("info", {}).get("props", {}).get("media.class")
        == "Audio/Source"
    ]


def find_base_source(sources: list[str]) -> str:
    config = {}
    try:
        config = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        pass

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
        raise RuntimeError("Select a physical source in ClearVoice before calibrating")
    return physical[0]


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


def capture_pass(base_source: str) -> dict[str, np.ndarray]:
    total_seconds = WARMUP_SECONDS + sum(duration for _, duration, _ in PHASES)
    expected_bytes = total_seconds * RATE * CHANNELS * 4
    sources = {"raw": base_source, "processed": "clearvoice_source"}
    processes = {
        label: subprocess.Popen(
            [
                "pw-record",
                f"--target={source}",
                f"--rate={RATE}",
                f"--channels={CHANNELS}",
                "--format=f32",
                "--raw",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=MANAGER_ENV,
        )
        for label, source in sources.items()
    }
    payloads = {}

    def reader(label: str):
        payloads[label] = read_exact(processes[label].stdout, expected_bytes)

    threads = [threading.Thread(target=reader, args=(label,)) for label in sources]
    for thread in threads:
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
        for thread in threads:
            thread.join(timeout=3)
        for process in processes.values():
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    captured = {}
    for label, payload in payloads.items():
        if len(payload) != expected_bytes:
            error = processes[label].stderr.read().decode(errors="replace").strip()
            raise RuntimeError(
                f"{label} capture returned {len(payload)} of {expected_bytes} bytes: {error}"
            )
        captured[label] = np.frombuffer(payload, dtype=np.float32).reshape(
            -1, CHANNELS
        )
    return captured


def band_energy(signal: np.ndarray) -> dict[str, float]:
    size = 4_096
    window = np.hanning(size).astype(np.float32)
    power = np.zeros(size // 2 + 1, dtype=np.float64)
    count = 0
    for start in range(0, len(signal) - size + 1, size):
        spectrum = np.fft.rfft(signal[start : start + size] * window)
        power += np.abs(spectrum) ** 2
        count += 1
    if not count or power.sum() == 0:
        return {name: -120.0 for name, _, _ in BANDS}

    frequencies = np.fft.rfftfreq(size, 1 / RATE)
    total = power[(frequencies >= 20) & (frequencies < 16_000)].sum()
    return {
        name: 10
        * math.log10(
            max(
                power[(frequencies >= low) & (frequencies < high)].sum()
                / total,
                1e-12,
            )
        )
        for name, low, high in BANDS
    }


def analyze(samples: np.ndarray) -> dict:
    result = {}
    cursor = WARMUP_SECONDS * RATE
    trim = RATE // 4
    for name, duration, _ in PHASES:
        block = samples[cursor + trim : cursor + duration * RATE - trim]
        cursor += duration * RATE
        channel_rms = np.sqrt(np.mean(block * block, axis=0))
        channel = np.nan_to_num(block[:, int(np.argmax(channel_rms))])
        rms = float(np.sqrt(np.mean(channel * channel)))
        peak = float(np.max(np.abs(channel)))
        result[name] = {
            "rms_dbfs": db(rms),
            "peak_dbfs": db(peak),
            "crest_db": db(peak / max(rms, 1e-12)),
            "clipped_samples": int(np.count_nonzero(np.abs(channel) >= 0.9999)),
            "bands_db_relative": band_energy(channel),
        }
    result["normal"]["snr_db"] = (
        result["normal"]["rms_dbfs"] - result["silence"]["rms_dbfs"]
    )
    return result


def print_report(report: dict):
    for path in ("raw", "processed"):
        print(f"\n{path.upper()}")
        for phase, _, _ in PHASES:
            values = report[path][phase]
            extra = f", SNR={values['snr_db']:.1f} dB" if phase == "normal" else ""
            print(
                f"  {phase:9} RMS {values['rms_dbfs']:6.1f} dBFS, "
                f"peak {values['peak_dbfs']:6.1f} dBFS, "
                f"clipped {values['clipped_samples']}{extra}"
            )
        bands = report[path]["normal"]["bands_db_relative"]
        print("  normal spectrum: " + ", ".join(f"{k} {v:.1f}" for k, v in bands.items()))


def save_results(base_source: str, passes: list[dict]):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_retained": False,
        "sample_rate": RATE,
        "channels": CHANNELS,
        "base_source": base_source,
        "passes": passes,
    }
    temporary = RESULTS_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(document, indent=2))
    temporary.replace(RESULTS_FILE)


def self_test():
    seconds = WARMUP_SECONDS + sum(duration for _, duration, _ in PHASES)
    timeline = np.arange(seconds * RATE, dtype=np.float32) / RATE
    mono = 0.1 * np.sin(2 * np.pi * 180 * timeline)
    samples = np.column_stack((mono, mono)).astype(np.float32)
    report = analyze(samples)
    assert -23.2 < report["normal"]["rms_dbfs"] < -22.8
    assert report["normal"]["clipped_samples"] == 0
    assert report["normal"]["bands_db_relative"]["body"] > -1
    print("Calibration self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    sources = list_sources()
    if "clearvoice_source" not in sources:
        raise SystemExit("ClearVoice must be running before calibration")
    base_source = find_base_source(sources)
    passes = []

    print("ClearVoice microphone calibration")
    print(f"Raw source: {base_source}")
    print("Audio stays in memory; only aggregate metrics are saved.")
    input("\nPress Enter to begin pass 1…")

    while True:
        number = len(passes) + 1
        print(f"\n=== PASS {number} ===", flush=True)
        captured = capture_pass(base_source)
        report = {label: analyze(samples) for label, samples in captured.items()}
        report["pass"] = number
        passes.append(report)
        save_results(base_source, passes)
        print_report(report)
        print(f"\nMetrics saved to {RESULTS_FILE}")
        if input("Another pass? [y/N] ").strip().lower() not in {"y", "yes"}:
            break

    print(f"\nDone. Tell me when to inspect {RESULTS_FILE}.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCalibration cancelled; no audio was saved.")
