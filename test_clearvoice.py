#!/usr/bin/env python3

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import clearvoice


def _config():
    return copy.deepcopy(clearvoice.DEFAULT_CONFIG)


def _node(name, node_id=1, client_id=7):
    props = {
        "node.name": name,
        "media.class": "Audio/Source",
        "client.id": client_id,
    }
    return {"id": node_id, "type": "PipeWire:Interface:Node", "info": {"props": props}}


def _client(pid, client_id=7):
    return {
        "id": client_id,
        "type": "PipeWire:Interface:Client",
        "info": {"props": {"application.process.id": pid}},
    }


def _output_port(node_id, channel):
    return {
        "type": "PipeWire:Interface:Port",
        "info": {
            "direction": "output",
            "props": {
                "node.id": node_id,
                "port.direction": "out",
                "audio.channel": channel,
            },
        },
    }


def test_clamp_percent():
    assert clearvoice._clamp_percent(-1, 70) == 0
    assert clearvoice._clamp_percent(101, 70) == 100
    assert clearvoice._clamp_percent("bad", 70) == 70


def test_mic_geometry_serializes_two_points_canonically():
    assert clearvoice.serialize_mic_geometry("-0.03, 0, 0, 0.03, 0, 0") == (
        "[[-0.03,0.0,0.0],[0.03,0.0,0.0]]"
    )
    assert "laptop-triple-linear" not in clearvoice.MIC_PRESETS
    assert clearvoice.DEFAULT_CONFIG["beamforming"]["preset"] == "laptop-dual-50mm"


def test_mic_geometry_rejects_invalid_or_coincident_points():
    for geometry in (
        "0,0,0,0,0,nan",
        "0,0,0,0,0,inf",
        "0,0,0,0,0",
        "0,0,0,0,0,0",
        "0,0,0,0.02,0,0,0.04,0,0",
    ):
        try:
            clearvoice.serialize_mic_geometry(geometry)
        except ValueError:
            continue
        raise AssertionError(f"invalid geometry accepted: {geometry}")


def test_echo_cancel_config_uses_modern_aec_and_beamformer_args():
    aec_conf = clearvoice._pw_conf_echo_cancel(monitor_mode=True)
    geometry = "-0.03,0,0,0.03,0,0"
    bf_conf = clearvoice._pw_conf_echo_cancel(
        beamforming=True, mic_geometry=geometry
    )

    for arg in (
        "webrtc.noise_suppression=false",
        "webrtc.high_pass_filter=false",
        "webrtc.gain_control=false",
        "webrtc.voice_detection=false",
        "webrtc.transient_suppression=false",
    ):
        assert arg in aec_conf
        assert arg in bf_conf
    assert "beamforming=0" not in aec_conf
    assert "webrtc.beamforming=true" in bf_conf
    assert f"webrtc.mic-geometry={clearvoice.serialize_mic_geometry(geometry)}" in bf_conf
    assert "webrtc.target-direction=[1.5707963,0,1]" in bf_conf
    for prop in (
        "audio.rate = 48000",
        "audio.channels = 2",
        "audio.position = [ FL FR ]",
        "stream.dont-remix = true",
    ):
        assert prop in bf_conf
    assert bf_conf.count("node.autoconnect = false") == 2


def test_pipewire_monitor_tracks_external_virtual_mic_links():
    callback = Mock()
    monitor = clearvoice.PipeWireMonitor(callback)
    objects = [
        _node("clearvoice_source", node_id=1),
        {
            "id": 2,
            "type": "PipeWire:Interface:Node",
            "info": {"props": {"node.name": "Recorder"}},
        },
        {
            "id": 3,
            "type": "PipeWire:Interface:Port",
            "info": {
                "props": {
                    "node.id": 1,
                    "port.direction": "out",
                    "audio.channel": "FL",
                }
            },
        },
        {
            "id": 4,
            "type": "PipeWire:Interface:Port",
            "info": {"props": {"node.id": 2, "port.direction": "in"}},
        },
        {
            "id": 5,
            "type": "PipeWire:Interface:Link",
            "info": {"output-port-id": 3, "input-port-id": 4},
        },
    ]
    with patch.object(clearvoice.GLib, "idle_add", side_effect=lambda fn, value: fn(value)):
        monitor._check_diff(json.dumps(objects).encode())
        assert monitor.nodes_active
        monitor._check_diff(
            json.dumps([{"id": 5, "type": None, "info": None}]).encode()
        )
        assert not monitor.nodes_active
    assert [item.args[0] for item in callback.call_args_list] == [True, False]


def test_echo_child_environment_ignores_parent_spa_path():
    with patch.dict(
        clearvoice.os.environ,
        {"SPA_PLUGIN_DIR": "/contaminated", "PIPEWIRE_MODULE_DIR": "/modules"},
        clear=False,
    ):
        aec_env = clearvoice.PipelineManager._echo_child_env(False)
        bf_env = clearvoice.PipelineManager._echo_child_env(True)

    assert aec_env["SPA_PLUGIN_DIR"] == str(clearvoice.SYSTEM_SPA_ROOT)
    assert bf_env["SPA_PLUGIN_DIR"] == (
        f"{clearvoice.PRIVATE_SPA_ROOT}:{clearvoice.SYSTEM_SPA_ROOT}"
    )
    assert aec_env["PIPEWIRE_MODULE_DIR"] == "/modules"
    assert bf_env["PIPEWIRE_MODULE_DIR"] == "/modules"


def test_beamformer_source_preflight_requires_manager_visible_fl_fr_ports():
    objects = [_node("mic"), _output_port(1, "FL"), _output_port(1, "FR")]
    with patch.object(clearvoice, "pw_dump_objects", return_value=objects) as dump:
        assert clearvoice.pw_source_has_separate_fl_fr("mic")
    dump.assert_called_once_with(manager=True)

    with patch.object(
        clearvoice, "pw_dump_objects", return_value=[_node("mic"), _output_port(1, "FL")]
    ):
        assert not clearvoice.pw_source_has_separate_fl_fr("mic")


def test_beamformer_waits_for_pid_owned_delayed_mono_port():
    client = _client(42)
    node = _node("beamformed")
    with (
        patch.object(
            clearvoice,
            "pw_dump_objects",
            side_effect=[[client, node], [client, node, _output_port(1, "MONO")]],
        ) as dump,
        patch.object(clearvoice.time, "sleep"),
    ):
        assert clearvoice.pw_wait_for_beamformed_source("beamformed", 42) == (True, "")
    assert dump.call_count == 2
    dump.assert_called_with(manager=True)


def test_beamformer_rejects_wrong_pid_stereo_or_multiple_ports():
    with patch.object(
        clearvoice,
        "pw_dump_objects",
        return_value=[_client(42), _node("beamformed", client_id=99)],
    ):
        ok, reason = clearvoice.pw_wait_for_beamformed_source("beamformed", 42)
    assert not ok
    assert "not owned" in reason

    for ports in (
        [_output_port(1, "FL"), _output_port(1, "FR")],
        [_output_port(1, "MONO"), _output_port(1, "AUX0")],
    ):
        with (
            patch.object(
                clearvoice,
                "pw_dump_objects",
                return_value=[_client(42), _node("beamformed"), *ports],
            ),
            patch.object(clearvoice.time, "monotonic", side_effect=[0, 0, 1]),
            patch.object(clearvoice.time, "sleep"),
        ):
            ok, reason = clearvoice.pw_wait_for_beamformed_source(
                "beamformed", 42, timeout=0.5
            )
        assert not ok
        assert "exactly one MONO" in reason


def test_beamformer_rejects_wrong_or_deleted_private_plugin_mapping():
    with tempfile.TemporaryDirectory() as directory:
        private_plugin = Path(directory) / "libspa-aec-webrtc.so"
        wrong_plugin = Path(directory) / "wrong.so"
        private_plugin.write_text("private")
        wrong_plugin.write_text("wrong")

        def map_line(plugin, deleted=False):
            suffix = " (deleted)" if deleted else ""
            return (
                f"7f000000-7f001000 r-xp 00000000 00:00 {plugin.stat().st_ino} "
                f"{plugin.resolve()}{suffix}\n"
            )

        with patch.object(Path, "read_text", return_value=map_line(private_plugin)):
            assert clearvoice.pw_private_aec_plugin_loaded(42, private_plugin)
        with patch.object(Path, "read_text", return_value=map_line(wrong_plugin)):
            assert not clearvoice.pw_private_aec_plugin_loaded(42, private_plugin)
        with patch.object(
            Path, "read_text", return_value=map_line(private_plugin, deleted=True)
        ):
            assert not clearvoice.pw_private_aec_plugin_loaded(42, private_plugin)


def test_private_deepfilter_requires_pinned_revision():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        plugin = root / "libdeep_filter_ladspa.so"
        revision = root / "libdeep_filter_ladspa.revision"
        plugin.write_text("plugin")
        with (
            patch.object(clearvoice, "PRIVATE_DEEPFILTER_PLUGIN", plugin),
            patch.object(clearvoice, "PRIVATE_DEEPFILTER_REVISION", revision),
        ):
            assert not clearvoice.private_deepfilter_ready()
            revision.write_text("wrong")
            assert not clearvoice.private_deepfilter_ready()
            revision.write_text(clearvoice.REQUIRED_DEEPFILTER_REVISION)
            assert clearvoice.private_deepfilter_ready()


def test_beamforming_preflight_failure_fails_open_without_changing_preferences():
    config = _config()
    config["noise_cancellation"]["enabled"] = False
    config["speaker_enhancement"]["enabled"] = False
    config["beamforming"]["enabled"] = True
    config["source_device"] = "mic"
    manager = clearvoice.PipelineManager(config)
    manager._started_once = True

    with tempfile.TemporaryDirectory() as directory:
        missing_plugin = Path(directory) / "missing.so"
        with (
            patch.object(clearvoice, "PRIVATE_AEC_PLUGIN", missing_plugin),
            patch.object(
                clearvoice,
                "pw_list_sources",
                return_value=[{"name": "mic", "id": 1, "description": "Mic"}],
            ),
            patch.object(clearvoice, "pactl_list_sinks", return_value=[]),
            patch.object(clearvoice, "pw_get_default_sink", return_value="sink"),
            patch.object(
                clearvoice,
                "pw_get_sink_active_port",
                return_value="analog-output-speaker",
            ),
            patch.object(clearvoice, "wp_set_setting", return_value=True) as setting,
            patch.object(clearvoice, "save_config"),
            patch.object(manager, "_kill_all") as kill_all,
            patch.object(manager, "_restore_previous_defaults") as restore_defaults,
            patch.object(manager, "_start_playback_output") as start_playback,
        ):
            ok, msg = manager.start()

    assert not ok
    assert "Private beamformer plugin not found" in msg
    assert not manager.running
    assert config["enabled"]
    assert config["beamforming"]["enabled"]
    assert ("clearvoice.lock-base-mic-audio", "false") in [
        call.args for call in setting.call_args_list
    ]
    kill_all.assert_called_once()
    restore_defaults.assert_called_once()
    start_playback.assert_not_called()


def test_active_port_detection_handles_missing_and_malformed_data():
    result = Mock(returncode=0, stderr="")
    result.stdout = json.dumps(
        [
            {"name": "other", "active_port": "analog-output-speaker"},
            {
                "name": "physical",
                "active_port": "analog-output-headphones",
            },
        ]
    )
    with patch.object(clearvoice.subprocess, "run", return_value=result):
        assert (
            clearvoice.pw_get_sink_active_port("physical")
            == "analog-output-headphones"
        )
        assert clearvoice.pw_get_sink_active_port("missing") is None

    result.stdout = json.dumps([{"name": "physical", "active_port": None}])
    with patch.object(clearvoice.subprocess, "run", return_value=result):
        assert clearvoice.pw_get_sink_active_port("physical") is None

    result.stdout = "not json"
    with patch.object(clearvoice.subprocess, "run", return_value=result):
        assert clearvoice.pw_get_sink_active_port("physical") is None


def test_headphone_mode_disables_effective_features_without_changing_preferences():
    config = _config()
    config["beamforming"]["enabled"] = True
    config["echo_cancellation"]["enabled"] = True
    config["speaker_enhancement"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    manager._headphone_mode = True

    assert not manager.bf_enabled
    assert not manager.aec_enabled
    assert not manager.spk_enabled
    assert manager.nc_enabled
    assert manager.studio_enabled
    assert config["beamforming"]["enabled"]
    assert config["echo_cancellation"]["enabled"]
    assert config["speaker_enhancement"]["enabled"]


def test_output_route_uses_headphone_or_speaker_gain():
    physical_sink = "alsa_output.physical"

    headphone = clearvoice.PipelineManager(_config())
    headphone._physical_sink = physical_sink
    headphone._headphone_mode = True
    with patch.object(headphone, "_set_playback_sink") as set_sink:
        headphone._start_playback_output()
    set_sink.assert_called_once_with(physical_sink, 100)

    speaker = clearvoice.PipelineManager(_config())
    speaker._physical_sink = physical_sink
    fake_proc = Mock()
    with tempfile.TemporaryDirectory() as runtime:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(clearvoice.subprocess, "Popen", return_value=fake_proc),
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "pw_wait_for_node", return_value=True),
            patch.object(speaker, "_set_playback_sink") as set_sink,
        ):
            speaker._start_playback_output()
    assert set_sink.call_args_list[-1].args == (clearvoice.SPEAKER_SINK_NAME, 80)


def test_playback_move_excludes_clearvoice_speaker_output():
    physical_sink = "alsa_output.physical"
    sink_inputs = [
        {"index": 1, "sink": 10, "properties": {"node.name": "Firefox"}},
        {
            "index": 2,
            "sink": 10,
            "properties": {"node.name": "clearvoice_speakers_out"},
        },
        {"index": 3, "sink": 20, "properties": {"node.name": "Other"}},
    ]
    moved = []

    def fake_run(command, **_kwargs):
        result = Mock(returncode=0, stderr="")
        if command[-1] == "sink-inputs":
            result.stdout = json.dumps(sink_inputs)
        else:
            result.stdout = ""
            moved.append(command)
        return result

    with (
        patch.object(
            clearvoice,
            "pactl_list_sinks",
            return_value=[
                {"index": 10, "name": physical_sink},
                {"index": 20, "name": clearvoice.SPEAKER_SINK_NAME},
            ],
        ),
        patch.object(clearvoice.subprocess, "run", side_effect=fake_run),
    ):
        assert clearvoice.pw_move_playback_streams(
            physical_sink, clearvoice.SPEAKER_SINK_NAME
        )

    assert moved == [
        ["pactl", "move-sink-input", "1", clearvoice.SPEAKER_SINK_NAME]
    ]


def test_unknown_route_keeps_last_confirmed_mode():
    manager = clearvoice.PipelineManager(_config())
    manager._physical_sink = "alsa_output.physical"
    manager._headphone_mode = True
    manager._route_confirmed = True

    with patch.object(clearvoice, "pw_get_sink_active_port", return_value="unknown-port"):
        manager._refresh_headphone_mode()

    assert manager.headphone_mode


def test_generated_configs_have_clearvoice_identity_and_restore_settings():
    filter_conf = clearvoice._pw_conf_filter_chain("/tmp/deepfilter.so")
    plain_filter_conf = clearvoice._pw_conf_filter_chain(
        "/tmp/deepfilter.so", studio_voice=False
    )
    echo_conf = clearvoice._pw_conf_echo_cancel(monitor_mode=True)
    intermediate_echo_conf = clearvoice._pw_conf_echo_cancel(is_intermediate=True)

    for conf in (filter_conf, echo_conf, intermediate_echo_conf):
        assert 'application.id = "org.clearvoice.ClearVoice"' in conf
        assert "clearvoice.client = true" in conf
    assert "state.restore-props = false" in filter_conf
    assert "monitor.mode = true" in echo_conf
    assert "state.restore-props = false" in echo_conf
    assert "state.restore-props = false" not in intermediate_echo_conf
    for conf in (filter_conf, plain_filter_conf):
        assert 'name = pretrim label = linear' in conf
        assert '"Mult" = 0.630957344' in conf
        assert 'name = restore label = linear' in conf
        assert '"Mult" = 1.584893192' in conf
        assert 'name   = agc' in conf
    assert 'name = hpf label = bq_highpass' in filter_conf
    assert 'name   = deesser' in filter_conf
    assert 'name   = limiter' in filter_conf
    assert '"attack"       = 0.5' in filter_conf
    assert 'name = hpf label = bq_highpass' not in plain_filter_conf
    assert 'name   = deesser' not in plain_filter_conf
    assert 'name   = limiter' not in plain_filter_conf


def test_policy_and_gain_setters_propagate_failures():
    manager = clearvoice.PipelineManager(_config())
    manager._base_mic_node = "mic"

    with (
        patch.object(clearvoice, "wp_set_setting", return_value=False),
        patch.object(clearvoice, "pw_set_node_volume", return_value=True),
    ):
        assert not manager._publish_lock_state(True)
        assert not manager._publish_base_mic_node("mic")
        assert not manager._publish_input_gain(70)
        assert not manager.set_input_gain(70)

    with (
        patch.object(clearvoice, "wp_set_setting", return_value=True),
        patch.object(clearvoice, "pw_set_node_volume", return_value=True),
        patch.object(clearvoice, "pw_node_exists", return_value=True),
    ):
        assert manager.set_input_gain(70)
        assert manager.set_output_gain(100)
        manager._running = True
        assert manager.set_lock_base_mic_audio(True)

    with patch.object(clearvoice, "wp_set_setting", return_value=False):
        manager._running = True
        assert not manager.set_lock_base_mic_audio(True)


def test_default_source_failure_unlocks_kills_and_restores():
    events = []

    class FakeProcess:
        def __init__(self):
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            events.append(("kill",))
            self.returncode = 0

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            events.append(("kill",))
            self.returncode = -9

    def fake_setting(key, value):
        events.append(("setting", key, value))
        return True

    def fake_set_default_source(name):
        events.append(("default", name))
        return name != clearvoice.VIRTUAL_MIC_NAME

    config = _config()
    config["speaker_enhancement"]["enabled"] = False
    manager = clearvoice.PipelineManager(config)
    manager._started_once = True

    with tempfile.TemporaryDirectory() as runtime:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(clearvoice, "find_ladspa_plugin", return_value="/tmp/plugin.so"),
            patch.object(
                clearvoice,
                "pw_list_sources",
                return_value=[{"name": "mic", "id": 1, "description": "Mic"}],
            ),
            patch.object(clearvoice, "pw_get_default_source", return_value="mic"),
            patch.object(clearvoice, "pactl_list_sinks", return_value=[]),
            patch.object(clearvoice, "pw_get_default_sink", return_value="sink"),
            patch.object(
                clearvoice,
                "pw_get_sink_active_port",
                return_value="analog-output-speaker",
            ),
            patch.object(manager, "_start_playback_output"),
            patch.object(clearvoice, "pw_wait_for_node", return_value=True),
            patch.object(clearvoice, "pw_node_exists", return_value=True),
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "wp_set_setting", side_effect=fake_setting),
            patch.object(
                clearvoice, "pw_set_default_source", side_effect=fake_set_default_source
            ),
            patch.object(clearvoice, "save_config"),
            patch.object(clearvoice.subprocess, "Popen", side_effect=lambda *args, **kwargs: FakeProcess()),
            patch.object(clearvoice.time, "sleep"),
        ):
            ok, msg = manager.start()

    assert not ok
    assert "Could not set default source" in msg
    assert not manager.running
    assert manager._base_mic_node is None

    failed_default = events.index(("default", clearvoice.VIRTUAL_MIC_NAME))
    unlock = next(
        index
        for index, event in enumerate(events[failed_default + 1 :], failed_default + 1)
        if event == ("setting", "clearvoice.lock-base-mic-audio", "false")
    )
    killed = events.index(("kill",))
    restored = next(
        index
        for index, event in enumerate(events[killed + 1 :], killed + 1)
        if event == ("default", "mic")
    )
    assert failed_default < unlock < killed < restored


def main():
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("ClearVoice tests passed")


if __name__ == "__main__":
    main()
