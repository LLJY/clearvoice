#!/usr/bin/env python3
"""ClearVoice Speaker Calibration Tool.

Plays a log sweep through speakers, records via the physical mic,
analyzes the frequency response, and outputs a correction EQ profile.

Usage:
    python3 calibrate.py [--volume 65] [--duration 5] [--output speaker_eq.json]

Requirements:
    - pipewire, pw-play, pw-record
    - numpy, scipy
    - Quiet room, moderate volume, don't move during sweep
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav


CONFIG_DIR = Path.home() / ".config" / "clearvoice"
DEFAULT_OUTPUT = CONFIG_DIR / "speaker_eq.json"


def find_physical_devices() -> tuple[str | None, str | None]:
    """Find the physical speaker sink and mic source."""
    r = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5)
    objects = json.loads(r.stdout)

    sink = source = None
    for obj in objects:
        props = obj.get("info", {}).get("props", {})
        mc = props.get("media.class", "")
        name = props.get("node.name", "")
        if name.startswith("clearvoice") or name.startswith("cv_"):
            continue
        if mc == "Audio/Sink" and not sink:
            sink = name
        elif mc == "Audio/Source" and not source:
            source = name
    return sink, source


def generate_sweep(duration: float, sample_rate: int = 48000) -> np.ndarray:
    """Generate a logarithmic sine sweep from 20Hz to 20kHz."""
    f1, f2 = 20, 20000
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    phase = (
        2
        * np.pi
        * f1
        * duration
        / np.log(f2 / f1)
        * (np.exp(t / duration * np.log(f2 / f1)) - 1)
    )
    sweep = 0.7 * np.sin(phase)

    # 0.5s silence padding
    silence = np.zeros(int(sample_rate * 0.5))
    return np.concatenate([silence, sweep, silence])


def record_sweep(
    sweep_path: str,
    mic_source: str,
    speaker_sink: str | None = None,
    sample_rate: int = 48000,
) -> str:
    """Play sweep and record simultaneously. Returns path to recording."""
    rec_path = tempfile.mktemp(suffix=".wav", prefix="cv_calib_")

    rec_cmd = [
        "pw-record",
        f"--target={mic_source}",
        f"--rate={sample_rate}",
        "--channels=1",
        "--format=f32",
        rec_path,
    ]
    play_cmd = ["pw-play"]
    if speaker_sink:
        play_cmd.append(f"--target={speaker_sink}")
    play_cmd.append(sweep_path)

    # Start recording, then play
    rec_proc = subprocess.Popen(rec_cmd)
    time.sleep(0.5)
    subprocess.run(play_cmd, check=True)
    time.sleep(0.5)
    rec_proc.terminate()
    rec_proc.wait(timeout=3)

    return rec_path


def analyze(sweep_path: str, recording_path: str, sample_rate: int = 48000) -> dict:
    """Analyze frequency response and generate correction profile."""
    _, ref = wav.read(sweep_path)
    _, rec = wav.read(recording_path)

    if rec.ndim > 1:
        rec = rec[:, 0]
    ref = ref.astype(np.float64)
    rec = rec.astype(np.float64)

    ml = min(len(ref), len(rec))
    ref, rec = ref[:ml], rec[:ml]

    freqs = np.fft.rfftfreq(ml, 1 / sample_rate)
    REF = np.fft.rfft(ref)
    REC = np.fft.rfft(rec)
    eps = np.max(np.abs(REF)) * 1e-6

    H = 20 * np.log10(np.abs(REC / (REF + eps)) + 1e-12)

    def avg(f):
        lo, hi = f / (2 ** (1 / 12)), f * (2 ** (1 / 12))
        m = (freqs >= lo) & (freqs <= hi)
        return float(np.mean(H[m])) if np.any(m) else 0.0

    ref_level = avg(1000)

    # Frequency response at key points
    response = {}
    check_freqs = [
        50,
        80,
        100,
        150,
        200,
        300,
        400,
        600,
        800,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        6000,
        8000,
        10000,
        12500,
        16000,
    ]
    print("\nSpeaker Frequency Response (normalized to 1kHz):")
    print(f"{'Freq':>8}  {'Level':>8}")
    print("─" * 20)
    for f in check_freqs:
        level = avg(f) - ref_level
        response[f] = round(level, 1)
        print(f"  {f:>6}  {level:>+6.1f} dB")

    # Flatness score
    test_freqs = np.geomspace(200, 10000, 30)
    vals = [avg(f) - ref_level for f in test_freqs]
    flatness = float(np.std(vals))
    print(f"\nFlatness (200Hz-10kHz): {flatness:.1f} dB std dev")

    return {
        "sample_rate": sample_rate,
        "reference_level_db": round(ref_level, 1),
        "response": response,
        "flatness_db": round(flatness, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="ClearVoice speaker calibration")
    parser.add_argument(
        "--volume",
        type=int,
        default=65,
        help="System volume %% during measurement (default: 65)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Sweep duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--sink",
        type=str,
        default=None,
        help="Target speaker sink (auto-detect if omitted)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Recording mic source (auto-detect if omitted)",
    )
    args = parser.parse_args()

    # Detect devices
    auto_sink, auto_source = find_physical_devices()
    sink = args.sink or auto_sink
    source = args.source or auto_source

    if not source:
        print("ERROR: No microphone found", file=sys.stderr)
        sys.exit(1)

    print(f"Speaker: {sink or '(default)'}")
    print(f"Mic:     {source}")
    print(f"Volume:  {args.volume}%")
    print(f"Sweep:   {args.duration}s")

    # Set volume
    subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{args.volume}%"],
        check=True,
    )

    # Generate sweep
    print("\nGenerating sweep...")
    sr = 48000
    sweep = generate_sweep(args.duration, sr)
    sweep_path = tempfile.mktemp(suffix=".wav", prefix="cv_sweep_")
    wav.write(sweep_path, sr, sweep.astype(np.float32))

    # Record
    print("Playing sweep — keep quiet and still...")
    rec_path = record_sweep(sweep_path, source, sink, sr)
    print("Done.")

    # Restore volume
    subprocess.run(
        ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "100%"],
    )

    # Analyze
    result = analyze(sweep_path, rec_path, sr)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_path}")

    # Cleanup
    Path(sweep_path).unlink(missing_ok=True)
    Path(rec_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
