#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "onnx==1.23.1",
#   "onnxruntime==1.29.0",
#   "numpy==2.5.3",
# ]
# ///
"""Build deterministic single-frame DeepFilterNet3-LL ONNX graphs."""

import argparse
import configparser
import copy
import hashlib
import io
import json
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, compose, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parents[2]
REV = "d375b2d8309e0935d165700c91da9de862a99c31"
ARCHIVE_SHA256 = "5998e58e8ba0e09bb76986ef97b84afa065a571ef282d4a1222f341e3251cf3a"
DEFAULT_OUT = ROOT / "plugin/models"
SR, FFT, HOP, NB_ERB, NB_DF = 48000, 960, 480, 32, 96
PARITY_LIMIT = 1e-4
WARMUP = 2 * SR // HOP
MEASURE = 2000
THRESHOLDS = (-15.0, 35.0, 35.0)
STAGES = ("enc", "erb_dec", "df_dec")
MAIN_OUTPUTS = {
    "enc": ("e0", "e1", "e2", "e3", "emb", "c0", "lsnr"),
    "erb_dec": ("m",),
    "df_dec": ("coefs", "302"),
}


def source_archive(path=None):
    if path is None:
        metadata = json.loads(subprocess.check_output([
            "cargo", "metadata", "--locked", "--format-version", "1",
            "--manifest-path", str(ROOT / "plugin/Cargo.toml"),
        ]))
        package = next(item for item in metadata["packages"] if item["name"] == "deep_filter")
        manifest = Path(package["manifest_path"])
        path = manifest.parent.parent / "models/DeepFilterNet3_ll_onnx.tar.gz"
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"unexpected DeepFilterNet3-LL archive sha256 {digest}: {path}; "
            f"expected {ARCHIVE_SHA256}"
        )
    return path


def extract_sources(destination, archive_path=None):
    """Extract only the three ONNX graphs and config from the hash-pinned tarball."""
    archive_path = source_archive(archive_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    wanted = {"enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"}
    extracted = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name not in wanted:
                continue
            if name in extracted or not member.isfile():
                raise RuntimeError(f"unexpected duplicate or non-file archive entry: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read archive entry: {member.name}")
            (destination / name).write_bytes(source.read())
            extracted.add(name)
    if extracted != wanted:
        raise RuntimeError(f"pinned model archive is missing: {sorted(wanted - extracted)}")

    cfg = configparser.ConfigParser()
    cfg.read(destination / "config.ini")
    expected = {
        "df": {"sr": "48000", "fft_size": "960", "hop_size": "480", "nb_erb": "32", "nb_df": "96", "df_order": "5"},
        "deepfilternet": {"conv_lookahead": "0", "emb_hidden_dim": "512", "df_hidden_dim": "512"},
    }
    for section, values in expected.items():
        for key, value in values.items():
            if cfg.get(section, key, fallback=None) != value:
                raise RuntimeError(f"unexpected pinned config value [{section}] {key}")
    if cfg.get("df", "norm_tau") != "1" or cfg.get("df", "df_lookahead") != "0":
        raise RuntimeError("unexpected normalization or DF lookahead in pinned config")
    return cfg, {stage: destination / f"{stage}.onnx" for stage in STAGES}


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def session(path_or_bytes):
    return ort.InferenceSession(
        str(path_or_bytes) if isinstance(path_or_bytes, Path) else path_or_bytes,
        session_options(),
        providers=["CPUExecutionProvider"],
    )


def staticize_time(model):
    for value in model.graph.input:
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_param == "S":
                dim.ClearField("dim_param")
                dim.dim_value = 1


def set_static_output_shapes(model):
    """Use ORT's concrete output shapes to remove exporter-only symbolic dimensions."""
    feeds = {}
    for value in model.graph.input:
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                raise RuntimeError(f"{value.name}: non-static input dimension {dim.dim_param!r}")
            shape.append(dim.dim_value)
        feeds[value.name] = np.zeros(shape, np.float32)
    outputs = session(model.SerializeToString()).run(None, feeds)
    if len(outputs) != len(model.graph.output):
        raise RuntimeError("ORT output count differs from ONNX graph output count")
    for value, output in zip(model.graph.output, outputs):
        del value.type.tensor_type.shape.dim[:]
        for size in output.shape:
            value.type.tensor_type.shape.dim.add().dim_value = int(size)


def normalize_model(model, graph_name):
    """Remove exporter/quantizer metadata and fix names that must be stable."""
    model.producer_name = "clearvoice.dfn3-ll.export"
    model.producer_version = "1"
    model.domain = ""
    model.model_version = 1
    model.doc_string = ""
    del model.metadata_props[:]
    model.graph.name = graph_name
    model.graph.doc_string = ""
    for node in model.graph.node:
        node.doc_string = ""
        for attribute in node.attribute:
            attribute.doc_string = ""
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        value.doc_string = ""
    for tensor in model.graph.initializer:
        tensor.doc_string = ""
    return model


def save_model(model, path, stage=None):
    normalize_model(model, f"dfn3_ll_{stage or Path(path).stem}")
    onnx.save_model(model, str(path))


def evaluate_pad_values(model):
    """Evaluate dynamic Pad vectors and their source tensor at S=1 via ORT."""
    probe = copy.deepcopy(model)
    pads = [node for node in probe.graph.node if node.op_type == "Pad"]
    rank = 4
    existing_outputs = {value.name for value in probe.graph.output}
    requested = []
    for node in pads:
        for name, dtype, shape in (
            (node.input[0], TensorProto.FLOAT, [None] * rank),
            (node.input[1], TensorProto.INT64, [2 * rank]),
        ):
            if name not in existing_outputs:
                probe.graph.output.append(helper.make_tensor_value_info(name, dtype, shape))
                existing_outputs.add(name)
            requested.append(name)
    feeds = {
        value.name: np.zeros(
            [dim.dim_value or 1 for dim in value.type.tensor_type.shape.dim], np.float32
        )
        for value in probe.graph.input
    }
    values = session(probe.SerializeToString()).run(requested, feeds)
    by_name = dict(zip(requested, values))
    observed = {}
    for node in pads:
        shape = by_name[node.input[0]].shape
        pads_value = np.asarray(by_name[node.input[1]], dtype=np.int64).reshape(-1)
        if pads_value.size != 2 * len(shape):
            raise RuntimeError(f"{node.name}: Pad vector rank does not match its input")
        before = int(pads_value[2])
        after = int(pads_value[2 + len(shape)])
        if before <= 0 or after != 0 or shape[2] != 1:
            raise RuntimeError(
                f"{node.name}: expected causal past-only time padding at axis 2; "
                f"shape={shape}, pads={pads_value.tolist()}"
            )
        observed[node.name] = (shape, pads_value)
    return observed


def prune_dead_nodes(graph):
    producers = {name: node for node in graph.node for name in node.output if name}
    live_values = {value.name for value in graph.output}
    live_nodes = set()
    pending = list(live_values)
    while pending:
        value = pending.pop()
        node = producers.get(value)
        if node is None or id(node) in live_nodes:
            continue
        live_nodes.add(id(node))
        for input_name in node.input:
            if input_name and input_name not in live_values:
                live_values.add(input_name)
                pending.append(input_name)
    kept = [node for node in graph.node if id(node) in live_nodes]
    del graph.node[:]
    graph.node.extend(kept)
    used = {name for node in graph.node for name in node.input if name}
    used.update(value.name for value in graph.output)
    kept_initializers = [init for init in graph.initializer if init.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)
    del graph.value_info[:]


def add_pad_state(model, node, observed, states):
    graph = model.graph
    source_shape, pads_value = observed[node.name]
    rank = len(source_shape)
    frames = int(pads_value[2])
    state_name = f"pad_{safe_name(node.name)}_state_in"
    output_name = f"pad_{safe_name(node.name)}_state_out"
    context_name = f"pad_{safe_name(node.name)}_context"
    state_shape = list(source_shape)
    state_shape[2] = frames
    graph.input.append(helper.make_tensor_value_info(state_name, TensorProto.FLOAT, state_shape))
    graph.output.append(helper.make_tensor_value_info(output_name, TensorProto.FLOAT, state_shape))

    stream_pads = pads_value.copy()
    stream_pads[2] = 0
    stream_pads[2 + rank] = 0
    pads_name = f"{context_name}_pads"
    graph.initializer.append(numpy_helper.from_array(stream_pads, pads_name))
    starts, ends, axes, steps = [f"{context_name}_{suffix}" for suffix in ("starts", "ends", "axes", "steps")]
    for name, value in ((starts, [1]), (ends, [2**63 - 1]), (axes, [2]), (steps, [1])):
        graph.initializer.append(numpy_helper.from_array(np.asarray(value, np.int64), name))
    pad_inputs = [context_name, pads_name]
    if len(node.input) > 2 and node.input[2]:
        pad_inputs.append(node.input[2])
    replacement = [
        helper.make_node(
            "Concat", [state_name, node.input[0]], [context_name],
            name=f"{context_name}_concat", axis=2,
        ),
        helper.make_node(
            "Pad", pad_inputs, [node.output[0]], name=f"{node.name}_stream_pad",
            mode=next(
                (helper.get_attribute_value(attr).decode() for attr in node.attribute if attr.name == "mode"),
                "constant",
            ),
        ),
        helper.make_node(
            "Slice", [context_name, starts, ends, axes, steps], [output_name],
            name=f"{context_name}_state_slice",
        ),
    ]
    states.append({"kind": "pad", "input": state_name, "output": output_name, "shape": state_shape})
    return replacement


def add_gru_state(graph, node, states):
    hidden = next(helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "hidden_size")
    state_in = f"gru_{safe_name(node.name)}_h_in"
    state_out = f"gru_{safe_name(node.name)}_h_out"
    shape = [1, 1, int(hidden)]
    graph.input.append(helper.make_tensor_value_info(state_in, TensorProto.FLOAT, shape))
    graph.output.append(helper.make_tensor_value_info(state_out, TensorProto.FLOAT, shape))
    node.input[5] = state_in
    node.output[1] = state_out
    states.append({"kind": "gru", "input": state_in, "output": state_out, "shape": shape})


def make_streaming_model(source_path, stage=None):
    stage = stage or Path(source_path).stem
    model = onnx.load(source_path)
    observed = evaluate_pad_values(model)
    staticize_time(model)
    states = []
    original_nodes = list(model.graph.node)
    replacements = {}
    for node in original_nodes:
        if node.op_type == "Pad":
            replacements[id(node)] = add_pad_state(model, node, observed, states)
        elif node.op_type == "GRU":
            add_gru_state(model.graph, node, states)
    new_nodes = []
    for node in original_nodes:
        new_nodes.extend(replacements.get(id(node), [node]))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    prune_dead_nodes(model.graph)
    set_static_output_shapes(model)
    normalize_model(model, f"dfn3_ll_{stage}")
    onnx.checker.check_model(model, full_check=True)
    return model, states


def constants(graph):
    values = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    values[node.output[0]] = numpy_helper.to_array(attr.t)
    return values


def add_gru_as_matmul(model, node, values):
    graph = model.graph
    inputs, outputs = list(node.input), list(node.output)
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    if attrs.get("direction", b"forward") not in (b"forward", "forward"):
        raise RuntimeError(f"{node.name}: only forward GRUs are supported")
    if attrs.get("linear_before_reset", 0) not in (0, 1):
        raise RuntimeError(f"{node.name}: unsupported linear_before_reset={attrs['linear_before_reset']}")
    w, r = np.asarray(values[inputs[1]], np.float32)[0], np.asarray(values[inputs[2]], np.float32)[0]
    hidden = int(attrs["hidden_size"])
    if len(inputs) > 3 and inputs[3]:
        bias = np.asarray(values[inputs[3]], np.float32)[0]
        wb, rb = bias[:3 * hidden], bias[3 * hidden:]
    else:
        wb = rb = np.zeros(3 * hidden, np.float32)
    if w.shape[0] != 3 * hidden or r.shape != (3 * hidden, hidden):
        raise RuntimeError(f"{node.name}: unexpected GRU weight shapes W={w.shape}, R={r.shape}")

    prefix = f"{safe_name(node.name)}_step"
    nodes = []

    def const(suffix, array):
        name = f"{prefix}_{suffix}"
        graph.initializer.append(numpy_helper.from_array(np.ascontiguousarray(array), name))
        return name

    def op(op_type, suffix, op_inputs, **attrs_):
        name = f"{prefix}_{suffix}"
        nodes.append(helper.make_node(op_type, op_inputs, [name], name=name, **attrs_))
        return name

    def linear(x, recurrent, gate, part):
        lo, hi = gate * hidden, (gate + 1) * hidden
        wm = const(f"{part}_w", w[lo:hi].T)
        rm = const(f"{part}_r", r[lo:hi].T)
        bw = const(f"{part}_wb", wb[lo:hi])
        br = const(f"{part}_rb", rb[lo:hi])
        xmat = op("MatMul", f"{part}_xmat", [x, wm])
        xlin = op("Add", f"{part}_xlin", [xmat, bw])
        hmat = op("MatMul", f"{part}_hmat", [recurrent, rm])
        hlin = op("Add", f"{part}_hlin", [hmat, br])
        return xlin, hlin

    x = op("Squeeze", "x_squeeze", [inputs[0]], axes=[0])
    h = op("Squeeze", "h_squeeze", [inputs[5]], axes=[0])
    xz, hz = linear(x, h, 0, "z")
    xr, hr = linear(x, h, 1, "r")
    zsum = op("Add", "z_sum", [xz, hz])
    rsum = op("Add", "r_sum", [xr, hr])
    z = op("Sigmoid", "z", [zsum])
    reset = op("Sigmoid", "r", [rsum])

    hstart, hend = 2 * hidden, 3 * hidden
    wh = const("h_w", w[hstart:hend].T)
    wbh = const("h_wb", wb[hstart:hend])
    xhmat = op("MatMul", "h_xmat", [x, wh])
    xh = op("Add", "h_xlin", [xhmat, wbh])
    rh = const("h_r", r[hstart:hend].T)
    rbh = const("h_rb", rb[hstart:hend])
    if attrs.get("linear_before_reset", 0) == 1:
        hmat = op("MatMul", "h_hmat", [h, rh])
        hlin = op("Add", "h_hlin", [hmat, rbh])
        reset_h = op("Mul", "h_reset", [reset, hlin])
    else:
        reset_state = op("Mul", "h_reset_state", [reset, h])
        hmat = op("MatMul", "h_hmat", [reset_state, rh])
        reset_h = op("Add", "h_reset", [hmat, rbh])
    candidate_sum = op("Add", "h_sum", [xh, reset_h])
    candidate = op("Tanh", "h_candidate", [candidate_sum])
    one = const("one", np.ones((1, hidden), np.float32))
    z_complement = op("Sub", "one_minus_z", [one, z])
    carry = op("Mul", "carry", [z, h])
    proposal = op("Mul", "proposal", [z_complement, candidate])
    hnew = op("Add", "h_new", [carry, proposal])
    nodes.append(helper.make_node("Unsqueeze", [hnew], [outputs[1]], name=f"{prefix}_h_out", axes=[0]))
    nodes.append(helper.make_node("Unsqueeze", [outputs[1]], [outputs[0]], name=f"{prefix}_sequence", axes=[0]))
    return nodes


def make_matmul_model(model, stage=None):
    model = copy.deepcopy(model)
    value_map = constants(model.graph)
    original_nodes = list(model.graph.node)
    replacements = {}
    for node in original_nodes:
        if node.op_type == "GRU":
            replacements[id(node)] = add_gru_as_matmul(model, node, value_map)
    new_nodes = []
    for node in original_nodes:
        new_nodes.extend(replacements.get(id(node), [node]))
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    prune_dead_nodes(model.graph)
    set_static_output_shapes(model)
    normalize_model(model, f"dfn3_ll_{stage or model.graph.name.removeprefix('dfn3_ll_')}")
    onnx.checker.check_model(model, full_check=True)
    return model


def make_fused(models):
    enc, erb, df = (copy.deepcopy(models[name]) for name in STAGES)
    enc_outputs = [value.name for value in enc.graph.output]
    erb_outputs = [value.name for value in erb.graph.output]
    df_outputs = [value.name for value in df.graph.output]
    first = compose.merge_models(
        enc,
        erb,
        [(name, name) for name in ("emb", "e3", "e2", "e1", "e0")],
        outputs=[*(f"enc/{name}" for name in enc_outputs), *(f"erb/{name}" for name in erb_outputs)],
        prefix1="enc/",
        prefix2="erb/",
    )
    fused = compose.merge_models(
        first,
        df,
        [("enc/emb", "emb"), ("enc/c0", "c0")],
        outputs=[*(value.name for value in first.graph.output), *(f"df/{name}" for name in df_outputs)],
        prefix2="df/",
    )
    expected = {f"enc/{name}" for name in MAIN_OUTPUTS["enc"]} | {"erb/m", "df/coefs", "df/302"}
    current = {output.name for output in fused.graph.output}
    if not expected <= current:
        raise RuntimeError(f"fused graph lost expected outputs: {sorted(expected - current)}")
    set_static_output_shapes(fused)
    normalize_model(fused, "dfn3_ll_fused")
    onnx.checker.check_model(fused, full_check=True)
    return fused


def graph_info(path):
    model = onnx.load(path)
    states = []
    outputs = {value.name: value for value in model.graph.output}
    for value in model.graph.input:
        if not value.name.endswith("_in"):
            continue
        state_out = value.name[:-3] + "_out"
        if state_out not in outputs:
            raise RuntimeError(f"missing paired state output for {value.name} in {Path(path).name}")
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in outputs[state_out].type.tensor_type.shape.dim]
        if shape != output_shape:
            raise RuntimeError(f"state shape mismatch: {value.name} {shape} != {state_out} {output_shape}")
        states.append({"input": value.name, "output": state_out, "shape": shape})
    return model, states


def stage_sessions(paths):
    return {stage: session(paths[stage]) for stage in STAGES}


def feat_sequence(frames):
    """Port DFState::analysis, feat_erb, and feat_cplx for mono 48 kHz."""
    rng = np.random.default_rng(0)
    count = frames * HOP
    t = np.arange(count, dtype=np.float64) / SR
    f0 = 140 + 60 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum(np.sin(h * phase) / h for h in range(1, 12))
    signal = (voice * (np.sin(2 * np.pi * 3 * t) > 0) * 0.06 + rng.normal(0, 0.03, count)).astype(np.float32)

    erb_scale = np.float32(np.float32(24.7) * np.float32(9.265))
    erb_low = np.float32(np.float32(9.265) * np.log1p(np.float32(0.0) / erb_scale))
    erb_high = np.float32(np.float32(9.265) * np.log1p(np.float32(SR // 2) / erb_scale))
    step = np.float32((erb_high - erb_low) / NB_ERB)
    widths = []
    prev, over = 0, 0
    freq_width = np.float32(SR) / np.float32(FFT)
    for i in range(1, NB_ERB + 1):
        erb_pos = np.float32(erb_low + np.float32(i) * step)
        hz = np.float32(erb_scale * np.expm1(erb_pos / np.float32(9.265)))
        fb = int(np.floor(np.float32(hz / freq_width) + np.float32(0.5)))
        width = fb - prev - over
        if width < 2:
            over = 2 - width
            width = 2
        else:
            over = 0
        widths.append(width)
        prev = fb
    widths[-1] += 1
    widths[-1] -= sum(widths) - (FFT // 2 + 1)
    if sum(widths) != FFT // 2 + 1:
        raise RuntimeError(f"ERB bank does not span the rFFT bins: {widths}")

    idx = np.arange(FFT, dtype=np.float64)
    wsin = np.sin(0.5 * np.pi * (idx + 0.5) / (FFT / 2))
    window = np.sin(0.5 * np.pi * wsin * wsin).astype(np.float32)
    norm = np.float32(1.0 / (FFT**2 / (2 * HOP)))
    dt = np.float32(HOP) / np.float32(SR)
    raw_alpha = np.float32(np.exp(-dt / np.float32(1.0)))
    alpha = np.float32(1.0)
    precision = 3
    while alpha >= 1:
        scale = np.float32(10**precision)
        alpha = np.float32(np.floor(np.float32(raw_alpha * scale) + np.float32(0.5)) / scale)
        precision += 1
    erb_step = np.float32(np.float32(-90.0 - -60.0) / np.float32(NB_ERB - 1))
    mean_state = np.asarray([np.float32(-60.0) + np.float32(i) * erb_step for i in range(NB_ERB)], np.float32)
    unit_step = np.float32(np.float32(0.0001 - 0.001) / np.float32(NB_DF - 1))
    unit_state = np.asarray([np.float32(0.001) + np.float32(i) * unit_step for i in range(NB_DF)], np.float32)
    previous = np.zeros(HOP, np.float32)
    feat_erb = np.empty((frames, NB_ERB), np.float32)
    feat_spec = np.empty((frames, 2, NB_DF), np.float32)
    for frame in range(frames):
        current = signal[frame * HOP:(frame + 1) * HOP]
        spectrum = (np.fft.rfft(np.concatenate((previous, current)) * window) * norm).astype(np.complex64)
        previous = current.copy()
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        start = 0
        bands = np.empty(NB_ERB, np.float32)
        for band, width in enumerate(widths):
            mean = np.float32(0.0)
            scale = np.float32(1.0 / width)
            for value in power[start:start + width]:
                mean = np.float32(mean + np.float32(value * scale))
            bands[band] = mean
            start += width
        erb_db = np.float32(10.0) * np.log10(bands + np.float32(1e-10)).astype(np.float32)
        mean_state[:] = erb_db * (np.float32(1.0) - alpha) + mean_state * alpha
        feat_erb[frame] = (erb_db - mean_state) / np.float32(40.0)

        z = spectrum[:NB_DF]
        magnitude = np.abs(z).astype(np.float32)
        unit_state[:] = magnitude * (np.float32(1.0) - alpha) + unit_state * alpha
        normalized = z / np.sqrt(unit_state).astype(np.float32)
        feat_spec[frame, 0] = normalized.real
        feat_spec[frame, 1] = normalized.imag
    return {
        "feat_erb": feat_erb[None, None, :, :],
        "feat_spec": feat_spec.transpose(1, 0, 2)[None, :, :, :],
        "alpha": float(alpha),
        "erb_widths": widths,
    }


def empty_states(states):
    return {state["input"]: np.zeros(state["shape"], np.float32) for state in states}


def run_sequence(sessions, state_lists, features, frames):
    collected = {stage: {name: [] for name in MAIN_OUTPUTS[stage]} for stage in STAGES}
    states = {stage: empty_states(state_lists[stage]) for stage in STAGES}
    for frame in range(frames):
        values = {}
        enc_feed = {
            "feat_erb": features["feat_erb"][:, :, frame:frame + 1, :],
            "feat_spec": features["feat_spec"][:, :, frame:frame + 1, :],
            **states["enc"],
        }
        result = dict(zip([item.name for item in sessions["enc"].get_outputs()], sessions["enc"].run(None, enc_feed)))
        for state in state_lists["enc"]:
            states["enc"][state["input"]] = result[state["output"]]
        values.update({name: result[name] for name in MAIN_OUTPUTS["enc"]})
        for stage, names in (("erb_dec", ("emb", "e3", "e2", "e1", "e0")), ("df_dec", ("emb", "c0"))):
            feed = {name: values[name] for name in names}
            feed.update(states[stage])
            stage_result = dict(zip([item.name for item in sessions[stage].get_outputs()], sessions[stage].run(None, feed)))
            for state in state_lists[stage]:
                states[stage][state["input"]] = stage_result[state["output"]]
            values.update({name: stage_result[name] for name in MAIN_OUTPUTS[stage]})
        for stage in STAGES:
            for name in MAIN_OUTPUTS[stage]:
                collected[stage][name].append(values[name])
    axes = {
        "enc": {name: (2 if name in {"e0", "e1", "e2", "e3", "c0"} else 1) for name in MAIN_OUTPUTS["enc"]},
        "erb_dec": {"m": 2},
        "df_dec": {"coefs": 1, "302": 1},
    }
    return {
        stage: {name: np.concatenate(items, axis=axes[stage][name]) for name, items in values.items()}
        for stage, values in collected.items()
    }


def run_fused(sess, states_meta, features, frames):
    model_outputs = [output.name for output in sess.get_outputs()]
    state = empty_states(states_meta)
    prefix = {"enc": "enc", "erb_dec": "erb", "df_dec": "df"}
    collected = {f"{prefix[stage]}/{name}": [] for stage in STAGES for name in MAIN_OUTPUTS[stage]}
    for frame in range(frames):
        feed = {
            "enc/feat_erb": features["feat_erb"][:, :, frame:frame + 1, :],
            "enc/feat_spec": features["feat_spec"][:, :, frame:frame + 1, :],
            **state,
        }
        result = dict(zip(model_outputs, sess.run(None, feed)))
        for item in states_meta:
            state[item["input"]] = result[item["output"]]
        for key in collected:
            collected[key].append(result[key])
    axes = {"e0": 2, "e1": 2, "e2": 2, "e3": 2, "c0": 2, "m": 2, "emb": 1, "lsnr": 1, "coefs": 1, "302": 1}
    return {
        stage: {
            name: np.concatenate(collected[f"{prefix[stage]}/{name}"], axis=axes[name])
            for name in MAIN_OUTPUTS[stage]
        }
        for stage in STAGES
    }


def compare_outputs(reference, actual, label):
    errors = {}
    worst = (0.0, None)
    for stage in STAGES:
        for name in MAIN_OUTPUTS[stage]:
            expected = reference[stage][name]
            result = actual[stage][name]
            if not np.isfinite(expected).all():
                raise RuntimeError(f"{label} {stage}.{name} reference has non-finite values")
            if not np.isfinite(result).all():
                raise RuntimeError(f"{label} {stage}.{name} actual has non-finite values")
            diff = np.abs(expected - result)
            maximum = float(diff.max(initial=0.0))
            errors[f"{stage}.{name}"] = maximum
            if maximum > worst[0]:
                worst = maximum, f"{stage}.{name}"
    print(f"{label}: max |diff|={worst[0]:.7g} at {worst[1]}")
    if worst[0] > PARITY_LIMIT:
        raise RuntimeError(f"{label} parity failed: max |diff|={worst[0]:.7g} > {PARITY_LIMIT}")
    return errors


def full_sequence_reference(originals, features, frames):
    enc_feed = {
        "feat_erb": features["feat_erb"][:, :, :frames, :],
        "feat_spec": features["feat_spec"][:, :, :frames, :],
    }
    outputs = {}
    outputs["enc"] = dict(zip(
        [value.name for value in originals["enc"].get_outputs()],
        originals["enc"].run(None, enc_feed),
    ))
    erb_feed = {name: outputs["enc"][name] for name in ("emb", "e3", "e2", "e1", "e0")}
    outputs["erb_dec"] = dict(zip(
        [value.name for value in originals["erb_dec"].get_outputs()],
        originals["erb_dec"].run(None, erb_feed),
    ))
    df_feed = {name: outputs["enc"][name] for name in ("emb", "c0")}
    outputs["df_dec"] = dict(zip(
        [value.name for value in originals["df_dec"].get_outputs()],
        originals["df_dec"].run(None, df_feed),
    ))
    return outputs


def reorder_reference(reference):
    return {stage: {name: reference[stage][name] for name in MAIN_OUTPUTS[stage]} for stage in STAGES}


def apply_stages(lsnr):
    minimum, max_erb, max_df = THRESHOLDS
    if lsnr < minimum:
        return False, True, False
    if lsnr > max_erb:
        return False, False, False
    if lsnr > max_df:
        return True, False, False
    return True, False, True


def quantization_metrics(fp32, quantized):
    ref_lsnr = fp32["enc"]["lsnr"].reshape(-1)
    q_lsnr = quantized["enc"]["lsnr"].reshape(-1)
    lsnr_max = float(np.max(np.abs(ref_lsnr - q_lsnr)))
    m_diff = quantized["erb_dec"]["m"] - fp32["erb_dec"]["m"]
    m_max = float(np.max(np.abs(m_diff)))
    m_rms = float(np.sqrt(np.mean(np.square(m_diff, dtype=np.float64))))
    coef_diff = quantized["df_dec"]["coefs"] - fp32["df_dec"]["coefs"]
    coef_ref = fp32["df_dec"]["coefs"]
    coef_rel_rms = float(
        np.sqrt(np.mean(np.square(coef_diff, dtype=np.float64)))
        / (np.sqrt(np.mean(np.square(coef_ref, dtype=np.float64))) + 1e-30)
    )
    ref_decisions = [apply_stages(float(value)) for value in ref_lsnr]
    quant_decisions = [apply_stages(float(value)) for value in q_lsnr]
    flips = float(np.mean([left != right for left, right in zip(ref_decisions, quant_decisions)]))
    return {
        "lsnr_max_abs_db": lsnr_max,
        "m_max_abs": m_max,
        "m_rms": m_rms,
        "coefs_relative_rms": coef_rel_rms,
        "apply_stages_flip_fraction": flips,
    }


def quantize_variant(input_paths, per_channel, suffix, out_dir=DEFAULT_OUT):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stage in STAGES:
        out = out_dir / f"{suffix}_{stage}.onnx"
        raw = out.with_name(f".{out.name}.quantizer.onnx")
        try:
            quantize_dynamic(
                str(input_paths[stage]),
                str(raw),
                op_types_to_quantize=["MatMul"],
                per_channel=per_channel,
                weight_type=QuantType.QInt8,
            )
            model = onnx.load(raw)
            set_static_output_shapes(model)
            normalize_model(model, f"dfn3_ll_{stage}")
            onnx.checker.check_model(model, full_check=True)
            save_model(model, out, stage)
        finally:
            raw.unlink(missing_ok=True)
        paths[stage] = out
    return paths


def parity_and_errors(source_paths, fp32_paths, matmul_paths=None, fused_path=None,
                      int8_paths=None, int8_pc_paths=None, frames=300):
    features = feat_sequence(frames)
    print(f"Features: {frames} frames, alpha={features['alpha']:.6f}, ERB bins={features['erb_widths']}")
    originals = {stage: session(source_paths[stage]) for stage in STAGES}
    reference = reorder_reference(full_sequence_reference(originals, features, frames))
    fp32_states = {stage: graph_info(fp32_paths[stage])[1] for stage in STAGES}
    fp32_sessions = stage_sessions(fp32_paths)
    fp32 = run_sequence(fp32_sessions, fp32_states, features, frames)
    parity = {"fp32-gru": compare_outputs(reference, fp32, "fp32-gru vs original full sequence")}

    if matmul_paths:
        matmul_states = {stage: graph_info(matmul_paths[stage])[1] for stage in STAGES}
        matmul = run_sequence(stage_sessions(matmul_paths), matmul_states, features, frames)
        parity["fp32-matmul"] = compare_outputs(fp32, matmul, "fp32-matmul vs fp32-gru")
    else:
        matmul = None
    if fused_path:
        fused_states = graph_info(fused_path)[1]
        fused = run_fused(session(fused_path), fused_states, features, frames)
        parity["fp32-fused"] = compare_outputs(fp32, fused, "fp32-fused vs fp32-gru")

    quant_errors = {}
    for label, paths in (("int8", int8_paths), ("int8-per-channel", int8_pc_paths)):
        if paths:
            state_lists = {stage: graph_info(paths[stage])[1] for stage in STAGES}
            values = run_sequence(stage_sessions(paths), state_lists, features, frames)
            quant_errors[label] = quantization_metrics(fp32, values)
            print(f"{label} error vs fp32-gru: " + json.dumps(quant_errors[label], sort_keys=True))

    return features, fp32_states, {
        "parity_max_abs": parity,
        "quantization_error": quant_errors,
        "features": {"frames": frames, "alpha": features["alpha"], "erb_widths": features["erb_widths"]},
    }


def interface_shape(value):
    shape = []
    for dim in value.type.tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            raise RuntimeError(f"{value.name}: non-static graph I/O dimension {dim.dim_param!r}")
        shape.append(int(dim.dim_value))
    return shape


def graph_schema(model, stage):
    inputs = {value.name: (value, interface_shape(value)) for value in model.graph.input}
    outputs = {value.name: (value, interface_shape(value)) for value in model.graph.output}
    required_inputs = {
        "enc": {"feat_erb", "feat_spec"},
        "erb_dec": {"emb", "e3", "e2", "e1", "e0"},
        "df_dec": {"emb", "c0"},
    }[stage]
    ordinary_inputs = {name for name in inputs if not name.endswith("_in")}
    if ordinary_inputs != required_inputs:
        raise RuntimeError(f"{stage}: unexpected non-state inputs {sorted(ordinary_inputs)}")
    if stage == "enc" and (inputs["feat_erb"][1] != [1, 1, 1, 32] or inputs["feat_spec"][1] != [1, 2, 1, 96]):
        raise RuntimeError("enc feature inputs do not match the fixed one-frame contract")
    expected_outputs = set(MAIN_OUTPUTS[stage])
    if stage == "df_dec" and "302" not in outputs:
        expected_outputs.remove("302")
    actual_outputs = {name for name in outputs if not name.endswith("_out")}
    if actual_outputs != expected_outputs:
        raise RuntimeError(
            f"{stage}: unexpected main outputs {sorted(actual_outputs)}; expected {sorted(expected_outputs)}"
        )

    states = []
    for name, (value, shape) in inputs.items():
        if not name.endswith("_in"):
            continue
        out_name = name[:-3] + "_out"
        if out_name not in outputs or outputs[out_name][1] != shape:
            raise RuntimeError(f"{stage}: invalid state pair {name} {shape} -> {out_name}")
        states.append({"input": name, "shape": shape, "output": out_name})
    state_outputs = {item["output"] for item in states}
    if state_outputs != {name for name in outputs if name.endswith("_out")}:
        raise RuntimeError(f"{stage}: graph has unpaired state outputs")
    return {
        "inputs": [{"name": name, "shape": shape} for name, (_, shape) in inputs.items()],
        "outputs": [{"name": name, "shape": shape} for name, (_, shape) in outputs.items()],
        "states": states,
    }


def validate_and_print_schemas(paths):
    schemas = {}
    for stage in STAGES:
        model = onnx.load(paths[stage])
        onnx.checker.check_model(model, full_check=True)
        if (
            model.producer_name != "clearvoice.dfn3-ll.export"
            or model.producer_version != "1"
            or model.domain
            or model.model_version != 1
            or model.graph.name != f"dfn3_ll_{stage}"
        ):
            raise RuntimeError(f"{paths[stage].name}: unstable producer metadata")
        if model.doc_string or model.graph.doc_string or model.metadata_props:
            raise RuntimeError(f"{paths[stage].name}: unexpected path-bearing metadata")
        if any(node.doc_string or any(attr.doc_string for attr in node.attribute) for node in model.graph.node):
            raise RuntimeError(f"{paths[stage].name}: unexpected node metadata")
        if any(value.doc_string for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)):
            raise RuntimeError(f"{paths[stage].name}: unexpected value metadata")
        if any(tensor.doc_string or tensor.data_location == TensorProto.EXTERNAL for tensor in model.graph.initializer):
            raise RuntimeError(f"{paths[stage].name}: external or metadata-bearing initializer")
        schemas[stage] = graph_schema(model, stage)
    print("Int8 graph I/O schema (static dimensions; state pairs are input -> output):")
    print(json.dumps(schemas, indent=2))
    return schemas


def export(out_dir=DEFAULT_OUT, archive_path=None):
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = source_archive(archive_path)
    with tempfile.TemporaryDirectory(prefix=".dfn3ll-export-", dir=out_dir) as temp:
        work = Path(temp)
        cfg, source_paths = extract_sources(work / "source", archive_path)
        gru_paths, fp32_paths, states = {}, {}, {}
        for stage in STAGES:
            model, states[stage] = make_streaming_model(source_paths[stage], stage)
            gru_paths[stage] = work / f"fp32-gru_{stage}.onnx"
            save_model(model, gru_paths[stage], stage)
            fp32_paths[stage] = work / f"dfn3_ll_fp32_{stage}.onnx"
            matmul = make_matmul_model(model, stage)
            save_model(matmul, fp32_paths[stage], stage)

        int8_paths = quantize_variant(fp32_paths, True, "dfn3_ll_int8", work)
        _, gru_states, report = parity_and_errors(
            source_paths, gru_paths, fp32_paths, int8_pc_paths=int8_paths, frames=300
        )
        for stage in STAGES:
            if graph_info(fp32_paths[stage])[1] != gru_states[stage]:
                raise RuntimeError(f"{fp32_paths[stage].name}: state schema differs from fp32-gru/{stage}")
            if graph_info(int8_paths[stage])[1] != gru_states[stage]:
                raise RuntimeError(f"{int8_paths[stage].name}: state schema differs from fp32-gru/{stage}")
        validate_and_print_schemas(int8_paths)

        print("Pinned source config: " + ", ".join(
            f"{key}={cfg.get(section, key)}"
            for section, key in (
                ("df", "fft_size"), ("df", "hop_size"), ("df", "nb_erb"), ("df", "nb_df"),
                ("df", "df_order"), ("df", "norm_tau"), ("deepfilternet", "conv_lookahead"),
                ("df", "df_lookahead"),
            )
        ))
        print("fp32-gru parity gate: " + json.dumps(report["parity_max_abs"]["fp32-gru"], sort_keys=True))
        print("fp32-matmul parity gate: " + json.dumps(report["parity_max_abs"]["fp32-matmul"], sort_keys=True))

        published = []
        for stage in STAGES:
            for paths in (fp32_paths, int8_paths):
                path = paths[stage]
                destination = out_dir / path.name
                # Each final file is replaced atomically after every parity/schema gate passes.
                import os
                os.replace(path, destination)
                published.append(destination)
        for path in published:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"{path.name}: sha256={digest} size={path.stat().st_size}")
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--archive", type=Path, help="DeepFilterNet3-LL source archive (default: cargo metadata dependency)")
    args = parser.parse_args()
    print(f"ONNX {onnx.__version__}, ONNX Runtime {ort.__version__}, source {REV}")
    print(f"Source archive sha256 pinned: {ARCHIVE_SHA256}")
    export(args.out, args.archive)


if __name__ == "__main__":
    main()
