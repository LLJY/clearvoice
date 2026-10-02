#!/usr/bin/env python3
import ctypes
import math
from pathlib import Path


class Descriptor(ctypes.Structure):
    pass


LadspaHandle = ctypes.c_void_p
Instantiate = ctypes.CFUNCTYPE(LadspaHandle, ctypes.POINTER(Descriptor), ctypes.c_ulong)
ConnectPort = ctypes.CFUNCTYPE(
    None, LadspaHandle, ctypes.c_ulong, ctypes.POINTER(ctypes.c_float)
)
Activate = ctypes.CFUNCTYPE(None, LadspaHandle)
Run = ctypes.CFUNCTYPE(None, LadspaHandle, ctypes.c_ulong)
Deactivate = ctypes.CFUNCTYPE(None, LadspaHandle)
Cleanup = ctypes.CFUNCTYPE(None, LadspaHandle)


class PortRangeHint(ctypes.Structure):
    _fields_ = [
        ("hint_descriptor", ctypes.c_int),
        ("lower_bound", ctypes.c_float),
        ("upper_bound", ctypes.c_float),
    ]


Descriptor._fields_ = [
    ("unique_id", ctypes.c_ulong),
    ("label", ctypes.c_char_p),
    ("properties", ctypes.c_int),
    ("name", ctypes.c_char_p),
    ("maker", ctypes.c_char_p),
    ("copyright", ctypes.c_char_p),
    ("port_count", ctypes.c_ulong),
    ("port_descriptors", ctypes.POINTER(ctypes.c_int)),
    ("port_names", ctypes.POINTER(ctypes.c_char_p)),
    ("port_range_hints", ctypes.POINTER(PortRangeHint)),
    ("implementation_data", ctypes.c_void_p),
    ("instantiate", Instantiate),
    ("connect_port", ConnectPort),
    ("activate", Activate),
    ("run", Run),
    ("run_adding", Run),
    ("set_run_adding_gain", ctypes.c_void_p),
    ("deactivate", Deactivate),
    ("cleanup", Cleanup),
]


def check_close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7), (
        actual,
        expected,
    )


def main():
    library_path = Path(__file__).resolve().parents[1] / "target/release/libclearvoice_ladspa.so"
    library = ctypes.CDLL(str(library_path))
    enumerate_descriptor = library.ladspa_descriptor
    enumerate_descriptor.argtypes = [ctypes.c_ulong]
    enumerate_descriptor.restype = ctypes.POINTER(Descriptor)

    descriptors = [enumerate_descriptor(index) for index in range(5)]
    assert all(descriptors[:4]), "a LADSPA descriptor is missing"
    assert not descriptors[4], "expected exactly four LADSPA labels"

    descriptor = descriptors[0].contents
    assert descriptor.label == b"clearvoice_dfn3_ll_mono"
    assert descriptor.name == b"ClearVoice DeepFilterNet3-LL (constant latency)"
    assert descriptor.maker == b"ClearVoice"
    assert descriptor.unique_id == 0x00C1EA03
    assert descriptor.port_count == 9
    names = [descriptor.port_names[index] for index in range(descriptor.port_count)]
    assert names == [
        b"Audio In",
        b"Audio Out",
        b"Latency (ms)",
        b"latency",
        b"Attenuation Limit (dB)",
        b"Min processing threshold (dB)",
        b"Max ERB processing threshold (dB)",
        b"Max DF processing threshold (dB)",
        b"Post Filter Beta",
    ]
    ports = [descriptor.port_descriptors[index] for index in range(descriptor.port_count)]
    assert ports == [9, 10, 5, 6, 5, 5, 5, 5, 5]
    hints = [descriptor.port_range_hints[index] for index in range(descriptor.port_count)]
    assert [hint.hint_descriptor for hint in hints] == [0, 0, 131, 0, 323, 67, 323, 323, 67]
    bounds = [(hint.lower_bound, hint.upper_bound) for hint in hints]
    expected_bounds = [
        (0.0, 0.0),
        (0.0, 0.0),
        (10.0, 200.0),
        (0.0, 0.0),
        (0.0, 100.0),
        (-15.0, 35.0),
        (-15.0, 35.0),
        (-15.0, 35.0),
        (0.0, 0.05),
    ]
    for actual, expected in zip(bounds, expected_bounds, strict=True):
        check_close(actual[0], expected[0])
        check_close(actual[1], expected[1])

    handle = descriptor.instantiate(descriptors[0], 44_100)
    if handle:
        descriptor.cleanup(handle)
    assert not handle, "non-48 kHz instantiate must fail"

    handle = descriptor.instantiate(descriptors[0], 48_000)
    assert handle, "48 kHz instantiate failed"
    input_buffer = (ctypes.c_float * 256)()
    output_buffer = (ctypes.c_float * 256)()
    latency_ms = ctypes.c_float(35.0)
    reported_latency = ctypes.c_float(-1.0)
    controls = [ctypes.c_float(value) for value in (100.0, -15.0, 35.0, 35.0, 0.0)]
    try:
        descriptor.connect_port(handle, 0, input_buffer)
        descriptor.connect_port(handle, 1, output_buffer)
        descriptor.connect_port(handle, 2, ctypes.byref(latency_ms))
        descriptor.connect_port(handle, 3, ctypes.byref(reported_latency))
        for index, control in enumerate(controls):
            descriptor.connect_port(handle, index + 4, ctypes.byref(control))
        descriptor.activate(handle)
        assert reported_latency.value == 1680.0, reported_latency.value

        outputs = []
        for block in range(16):
            for index in range(len(input_buffer)):
                sample = block * len(input_buffer) + index
                input_buffer[index] = 0.3 * math.sin(2.0 * math.pi * 440.0 * sample / 48_000)
            descriptor.run(handle, len(input_buffer))
            outputs.extend(output_buffer)
        assert all(math.isfinite(sample) for sample in outputs)
        assert any(abs(sample) > 1e-6 for sample in outputs), "audio output stayed silent"
    finally:
        descriptor.deactivate(handle)
        descriptor.cleanup(handle)

    base_names = [b"Audio In", b"Audio Out", b"Latency (ms)", b"latency"]
    for index, (label, name, unique_id) in enumerate(
        [
            (
                b"clearvoice_fastenhancer_b_mono",
                b"ClearVoice FastEnhancer-B (constant latency)",
                0x00C1EA04,
            ),
            (
                b"clearvoice_fastenhancer_s_mono",
                b"ClearVoice FastEnhancer-S (constant latency)",
                0x00C1EA05,
            ),
            (
                b"clearvoice_fastenhancer_m_mono",
                b"ClearVoice FastEnhancer-M (constant latency)",
                0x00C1EA06,
            ),
        ],
        start=1,
    ):
        descriptor_ptr = descriptors[index]
        descriptor = descriptor_ptr.contents
        assert descriptor.label == label
        assert descriptor.name == name
        assert descriptor.unique_id == unique_id
        assert descriptor.port_count == 4
        assert [descriptor.port_names[port] for port in range(4)] == base_names
        assert [descriptor.port_descriptors[port] for port in range(4)] == [9, 10, 5, 6]
        assert [descriptor.port_range_hints[port].hint_descriptor for port in range(4)] == [
            0,
            0,
            131,
            0,
        ]

        handle = descriptor.instantiate(descriptor_ptr, 48_000)
        assert handle, f"{label.decode()} instantiate failed"
        input_buffer = (ctypes.c_float * 256)()
        output_buffer = (ctypes.c_float * 256)()
        latency_ms = ctypes.c_float(35.0)
        reported_latency = ctypes.c_float(-1.0)
        outputs = []
        try:
            descriptor.connect_port(handle, 0, input_buffer)
            descriptor.connect_port(handle, 1, output_buffer)
            descriptor.connect_port(handle, 2, ctypes.byref(latency_ms))
            descriptor.connect_port(handle, 3, ctypes.byref(reported_latency))
            descriptor.activate(handle)
            assert reported_latency.value == 1680.0, reported_latency.value
            for block in range(16):
                for sample_index in range(len(input_buffer)):
                    sample = block * len(input_buffer) + sample_index
                    input_buffer[sample_index] = 0.3 * math.sin(
                        2.0 * math.pi * 440.0 * sample / 48_000
                    )
                descriptor.run(handle, len(input_buffer))
                outputs.extend(output_buffer)
            assert all(math.isfinite(sample) for sample in outputs)
            assert any(abs(sample) > 1e-6 for sample in outputs), "audio output stayed silent"
        finally:
            descriptor.deactivate(handle)
            descriptor.cleanup(handle)

    print(
        "ABI smoke passed: 4 labels (DFN 9 ports, FastEnhancer-B/S/M 4 ports), "
        "44.1 kHz rejected, latency=1680 samples before run, 16 finite sine blocks per label."
    )


if __name__ == "__main__":
    main()
