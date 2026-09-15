#!/usr/bin/env python3

import copy
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import clearvoice


def _config():
    return copy.deepcopy(clearvoice.DEFAULT_CONFIG)


def test_clamp_percent():
    assert clearvoice._clamp_percent(-1, 70) == 0
    assert clearvoice._clamp_percent(101, 70) == 100
    assert clearvoice._clamp_percent("bad", 70) == 70


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
