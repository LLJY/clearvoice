#!/usr/bin/env python3

import copy
import hashlib
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import Mock, call, patch

import clearvoice


def _config():
    return copy.deepcopy(clearvoice.DEFAULT_CONFIG)


def _bare_tray(manager):
    tray = object.__new__(clearvoice.ClearVoiceTray)
    tray.pipeline = manager
    tray.config = _config()
    tray._pw_monitor = None
    tray._route_probe_pending = False
    tray._route_restart_pending = False
    tray._health_restart_pending = False
    tray._quitting = False
    tray._enable_converge_lock = threading.Lock()
    tray._enable_converge_running = False
    tray._enable_replay_mic = False
    tray._enable_converge_thread = None
    tray._mi_enable = Mock()
    tray._update_icon = Mock()
    tray._update_status = Mock()
    return tray


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


def _managed_link_graph(linked=(), include_filter=False, filter_linked=()):
    objects = [_node("mic", 1), _node("clearvoice_ec_capture", 2)]
    for node_id, names, direction, first_id in (
        (1, ("capture_FL", "capture_FR"), "out", 10),
        (2, ("input_FL", "input_FR"), "in", 20),
    ):
        for index, name in enumerate(names):
            channel = name.rsplit("_", 1)[-1]
            objects.append(
                {
                    "id": first_id + index,
                    "type": "PipeWire:Interface:Port",
                    "info": {
                        "direction": direction,
                        "props": {
                            "node.id": node_id,
                            "port.name": name,
                            "port.direction": direction,
                            "audio.channel": channel,
                        },
                    },
                }
            )
    links = []
    for index, channel in enumerate(linked):
        port_index = 0 if channel == "FL" else 1
        links.append((10 + port_index, 20 + port_index))
    if include_filter:
        objects.extend((_node(clearvoice.EC_SOURCE_NAME, 3), _node("clearvoice_capture", 4)))
        objects.extend(
            (
                {
                    "id": 30,
                    "type": "PipeWire:Interface:Port",
                    "info": {
                        "direction": "out",
                        "props": {
                            "node.id": 3,
                            "port.name": "capture_MONO",
                            "port.direction": "out",
                        },
                    },
                },
                {
                    "id": 40,
                    "type": "PipeWire:Interface:Port",
                    "info": {
                        "direction": "in",
                        "props": {
                            "node.id": 4,
                            "port.name": "input_MONO",
                            "port.direction": "in",
                        },
                    },
                },
            )
        )
        if "MONO" in filter_linked:
            links.append((30, 40))
    for index, (output_id, input_id) in enumerate(links):
        objects.append(
            {
                "id": 50 + index,
                "type": "PipeWire:Interface:Link",
                "info": {"output-port-id": output_id, "input-port-id": input_id},
            }
        )
    return objects


def test_repeated_health_failures_queue_one_restart_until_it_finishes():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    tray = _bare_tray(manager)
    release = threading.Event()
    restarts = []

    def restart():
        restarts.append(1)
        assert release.wait(2)
        return True, "restarted"

    with (
        patch.object(manager, "check_health", return_value=False),
        patch.object(manager, "restart", side_effect=restart),
        patch.object(clearvoice.GLib, "idle_add"),
    ):
        for _ in range(4):  # liveness ticks while the first restart is blocked
            tray._on_health_tick()
        time.sleep(0.1)
        assert len(restarts) == 1 and tray._health_restart_pending
        release.set()
        deadline = time.monotonic() + 2
        while tray._health_restart_pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not tray._health_restart_pending
        tray._on_health_tick()  # a later failure may restart again
        deadline = time.monotonic() + 2
        while len(restarts) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert len(restarts) == 2


def test_orphan_cleanup_only_matches_our_pipewire_children():
    import re

    pattern = re.compile(clearvoice.ORPHAN_PROCESS_PATTERN)
    assert pattern.search("pipewire -c /run/user/1000/clearvoice/filter-chain.conf")
    assert pattern.search(
        "pipewire -c /home/u/.local/share/clearvoice/speaker-chain.conf"
    )
    # A shell or editor whose command line merely mentions the path must survive.
    assert not pattern.search(
        "/bin/zsh -c pgrep -f '^pipewire -c' /run/user/1000/clearvoice/x.conf"
    )
    assert not pattern.search("nvim /run/user/1000/clearvoice/filter-chain.conf")
    with patch.object(clearvoice.subprocess, "run") as run, patch.object(
        clearvoice.os, "kill"
    ):
        run.return_value = Mock(stdout="")
        clearvoice.PipelineManager._cleanup_orphans()
    command = run.call_args.args[0]
    assert command[:3] == ["pgrep", "-u", str(clearvoice.os.getuid())]
    assert command[-1] == clearvoice.ORPHAN_PROCESS_PATTERN


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


def test_pipewire_monitor_requests_reconcile_only_for_active_link_removal():
    def graph(target_name):
        return [
            _node("clearvoice_source", 1),
            _node(target_name, 2),
            {
                "id": 3,
                "type": "PipeWire:Interface:Port",
                "info": {"props": {"node.id": 1, "port.direction": "out"}},
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

    removed = [{"id": 5, "type": None, "info": None}]
    active_reconcile = Mock()
    active = clearvoice.PipeWireMonitor(
        Mock(), on_link_removed=active_reconcile
    )
    standby_reconcile = Mock()
    standby = clearvoice.PipeWireMonitor(
        Mock(), on_link_removed=standby_reconcile
    )
    with patch.object(clearvoice.GLib, "idle_add"):
        active._check_diff(json.dumps(graph("Recorder")).encode())
        assert active.nodes_active
        active._check_diff(json.dumps(removed).encode())

        standby._check_diff(json.dumps(graph("clearvoice_capture")).encode())
        assert not standby.nodes_active
        standby._check_diff(json.dumps(removed).encode())

    active_reconcile.assert_called_once_with()
    standby_reconcile.assert_not_called()


def test_process_health_and_link_reconcile_timers_have_separate_cadences():
    tray = _bare_tray(Mock())
    with patch.object(clearvoice.GLib, "timeout_add_seconds") as timeout:
        tray._register_periodic_checks()
    assert timeout.call_args_list == [
        call(2, tray._on_health_tick),
        call(10, tray._on_reconcile_tick),
    ]


def test_deepfilter_worker_detection_and_rtkit_promotion():
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    ps_result = Mock(
        returncode=0,
        stdout="100 pipewire TS\n101 module-rt TS\n102 data-loop.0 RR\n103 pipewire TS\n",
    )
    rtkit_result = Mock(returncode=0, stderr="")
    with patch.object(
        clearvoice.subprocess,
        "run",
        side_effect=[ps_result, rtkit_result, ps_result],
    ) as run, patch.object(
        clearvoice.os,
        "sched_getscheduler",
        # RTKit reports SCHED_RR with SCHED_RESET_ON_FORK set.
        side_effect=[
            clearvoice.os.SCHED_OTHER,
            clearvoice.os.SCHED_RR | clearvoice.os.SCHED_RESET_ON_FORK,
        ],
    ), patch.object(
        clearvoice.os,
        "sched_getparam",
        side_effect=[Mock(sched_priority=0), Mock(sched_priority=10)],
    ), patch.object(
        clearvoice,
        "_deepfilter_worker_runtime_ns",
        side_effect=[0, 1_000_000],
    ), patch.object(
        clearvoice.time, "monotonic", side_effect=lambda: clock[0]
    ), patch.object(
        clearvoice.time, "sleep", side_effect=sleep
    ):
        assert clearvoice.promote_deepfilter_worker(100) == 103
    rtkit_call = next(
        call for call in run.call_args_list if call.args[0][0] == "busctl"
    )
    assert rtkit_call.args[0][-3:] == ["100", "103", "10"]


def test_deepfilter_accepts_verified_rt_worker_without_repromoting():
    ps_result = Mock(
        returncode=0, stdout="100 pipewire TS\n101 pipewire RR\n"
    )
    with (
        patch.object(clearvoice.subprocess, "run", return_value=ps_result) as run,
        patch.object(
            clearvoice.os,
            "sched_getscheduler",
            return_value=clearvoice.os.SCHED_RR | clearvoice.os.SCHED_RESET_ON_FORK,
        ),
        patch.object(
            clearvoice.os, "sched_getparam", return_value=Mock(sched_priority=10)
        ),
        patch.object(clearvoice.log, "debug") as debug,
    ):
        assert clearvoice.promote_deepfilter_worker(100) == 101
    assert run.call_count == 1
    assert run.call_args.args[0][0] == "ps"
    debug.assert_called_once_with(
        "DeepFilter worker %d already uses SCHED_RR %d", 101, 10
    )


def test_deepfilter_retries_worker_discovery_and_excludes_graph_threads():
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    empty = Mock(
        returncode=0,
        stdout="100 pipewire TS\n101 module-rt RR\n102 data-loop.0 TS\n",
    )
    worker = Mock(
        returncode=0,
        stdout=(
            "100 pipewire TS\n101 module-rt RR\n"
            "102 data-loop.0 TS\n103 pipewire TS\n"
        ),
    )
    rtkit_result = Mock(returncode=0, stderr="")
    with (
        patch.object(
            clearvoice.subprocess,
            "run",
            side_effect=[empty, worker, rtkit_result, worker],
        ),
        patch.object(clearvoice.time, "monotonic", side_effect=lambda: clock[0]),
        patch.object(clearvoice.time, "sleep", side_effect=sleep),
        patch.object(
            clearvoice,
            "_deepfilter_worker_runtime_ns",
            side_effect=[0, 1_000_000],
        ),
        patch.object(
            clearvoice.os,
            "sched_getscheduler",
            side_effect=[clearvoice.os.SCHED_OTHER, clearvoice.os.SCHED_RR],
        ),
        patch.object(
            clearvoice.os,
            "sched_getparam",
            side_effect=[Mock(sched_priority=0), Mock(sched_priority=10)],
        ),
    ):
        assert clearvoice.promote_deepfilter_worker(100, timeout=1) == 103


def test_named_clearvoice_worker_is_promoted_without_the_idle_gate():
    worker = Mock(returncode=0, stdout="100 pipewire TS\n104 cv-dsp-worker TS\n")
    rtkit_result = Mock(returncode=0, stderr="")
    with (
        patch.object(
            clearvoice.subprocess, "run", side_effect=[worker, rtkit_result, worker]
        ) as run,
        patch.object(clearvoice, "_deepfilter_worker_runtime_ns") as schedstat,
        patch.object(
            clearvoice.os,
            "sched_getscheduler",
            side_effect=[clearvoice.os.SCHED_OTHER, clearvoice.os.SCHED_RR],
        ),
        patch.object(
            clearvoice.os,
            "sched_getparam",
            side_effect=[Mock(sched_priority=0), Mock(sched_priority=10)],
        ),
    ):
        assert (
            clearvoice.promote_deepfilter_worker(100, worker_name="cv-dsp-worker")
            == 104
        )
    schedstat.assert_not_called()
    assert "MakeThreadRealtimeWithPID" in run.call_args_list[1].args[0]


def test_deepfilter_waits_for_idle_schedstat_window_before_rtkit_promotion():
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    ps_result = Mock(returncode=0, stdout="100 pipewire TS\n101 pipewire TS\n")
    rtkit_result = Mock(returncode=0, stderr="")
    with (
        patch.object(
            clearvoice.subprocess,
            "run",
            side_effect=[ps_result, ps_result, rtkit_result, ps_result],
        ) as run,
        patch.object(
            clearvoice.os,
            "sched_getscheduler",
            side_effect=[
                clearvoice.os.SCHED_OTHER,
                clearvoice.os.SCHED_OTHER,
                clearvoice.os.SCHED_RR,
            ],
        ),
        patch.object(
            clearvoice.os,
            "sched_getparam",
            side_effect=[
                Mock(sched_priority=0),
                Mock(sched_priority=0),
                Mock(sched_priority=10),
            ],
        ),
        patch.object(
            clearvoice,
            "_deepfilter_worker_runtime_ns",
            side_effect=[0, 180_000_000, 180_000_000, 280_000_000],
        ),
        patch.object(clearvoice.time, "monotonic", side_effect=lambda: clock[0]),
        patch.object(clearvoice.time, "sleep", side_effect=sleep),
    ):
        assert clearvoice.promote_deepfilter_worker(100) == 101

    assert sum(call.args[0][0] == "busctl" for call in run.call_args_list) == 1


def test_deepfilter_skips_rt_promotion_when_schedstat_is_unreadable():
    ps_result = Mock(returncode=0, stdout="100 pipewire TS\n101 pipewire TS\n")
    with (
        patch.object(clearvoice.subprocess, "run", return_value=ps_result) as run,
        patch.object(
            clearvoice.os, "sched_getscheduler", return_value=clearvoice.os.SCHED_OTHER
        ),
        patch.object(
            clearvoice.os, "sched_getparam", return_value=Mock(sched_priority=0)
        ),
        patch.object(clearvoice, "_deepfilter_worker_runtime_ns", return_value=None),
        patch.object(clearvoice.log, "debug") as debug,
    ):
        assert clearvoice.promote_deepfilter_worker(100) is None

    assert run.call_count == 1
    assert run.call_args.args[0][0] == "ps"
    debug.assert_called_once()


def test_deepfilter_promotion_covers_speaker_nc_only_and_headphones():
    cases = []
    speaker = _config()
    speaker["beamforming"]["enabled"] = True
    cases.append((speaker, False))
    nc_only = _config()
    nc_only["speaker_enhancement"]["enabled"] = False
    cases.append((nc_only, False))
    headphones = _config()
    cases.append((headphones, True))

    for config, headphone_mode in cases:
        manager = clearvoice.PipelineManager(config)
        manager._running = True
        manager._headphone_mode = headphone_mode
        manager._mic_demand = True
        manager._fc_proc = Mock(pid=42, poll=Mock(return_value=None))
        with patch.object(
            clearvoice, "promote_deepfilter_worker", return_value=73
        ) as promote:
            manager._reconcile_deepfilter(True, 0, 0)
        promote.assert_called_once()
        assert not promote.call_args.kwargs["should_cancel"]()


def test_deepfilter_rechecks_live_worker_and_cancels_stale_generation():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    manager._mic_demand = True
    manager._fc_proc = Mock(pid=42, poll=Mock(return_value=None))
    with patch.object(clearvoice, "promote_deepfilter_worker", return_value=72) as promote:
        manager._reconcile_deepfilter(True, 0, 0)
    promote.assert_called_once()

    def stale_completion(_pid, should_cancel=None):
        manager._generation += 1
        assert should_cancel()
        return 73

    with patch.object(
        clearvoice, "promote_deepfilter_worker", side_effect=stale_completion
    ) as promote:
        manager._reconcile_deepfilter(True, 0, 0)
    promote.assert_called_once()


def test_active_mic_intent_is_reconciled_after_start_and_restart():
    manager = _active_beamforming_manager()
    manager._running = False
    manager._mic_demand = False
    manager._demand_revision = 0
    links = []

    def start_pipeline():
        manager._generation += 1
        manager._running = True
        return True, "started"

    def stop_pipeline():
        manager._generation += 1
        manager._running = False
        return True, "stopped"

    with (
        patch.object(clearvoice, "pw_dump_objects", return_value=_managed_link_graph()),
        patch.object(
            clearvoice,
            "pw_link_ports",
            side_effect=lambda *args: links.append(args) or True,
        ),
    ):
        assert manager.set_mic_demand(True)
        manager._reconcile_thread.join(2)
        with patch.object(manager, "_start_locked", side_effect=start_pipeline):
            assert manager.start()[0]
            manager._reconcile_thread.join(2)

        manager._running = True
        with (
            patch.object(manager, "_stop_locked", side_effect=stop_pipeline),
            patch.object(manager, "_start_locked", side_effect=start_pipeline),
            patch.object(clearvoice.time, "sleep"),
        ):
            assert manager.restart()[0]
            manager._reconcile_thread.join(2)

    assert links == [
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", True),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", True),
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", True),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", True),
    ]
    assert manager._mic_demand


def test_disable_returns_while_pipeline_lifecycle_lock_is_busy():
    config = _config()
    config["noise_cancellation"]["enabled"] = False
    config["beamforming"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    tray = _bare_tray(manager)
    manager._running = True
    manager._base_mic_node = "mic"
    item = Mock()
    item.get_active.return_value = False
    query_entered = threading.Event()
    release = threading.Event()
    callback_done = threading.Event()
    real_thread = threading.Thread
    spawned = []

    def blocked_dump(**_kwargs):
        query_entered.set()
        assert release.wait(2)
        return _managed_link_graph()

    def worker_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        spawned.append(thread)
        return thread

    with (
        patch.object(clearvoice.threading, "Thread", side_effect=worker_thread),
        patch.object(clearvoice, "save_config"),
        patch.object(clearvoice.GLib, "idle_add") as idle_add,
        patch.object(clearvoice, "pw_dump_objects", side_effect=blocked_dump),
        patch.object(clearvoice, "pw_link_ports", return_value=True),
        patch.object(manager, "_publish_lock_state", return_value=True),
        patch.object(manager, "_publish_output_volume_lock", return_value=True),
        patch.object(manager, "_restore_previous_defaults"),
    ):
        assert manager.set_mic_demand(True)
        assert query_entered.wait(1)
        callback = real_thread(
            target=lambda: (tray._on_enable(item), callback_done.set())
        )
        callback.start()
        assert callback_done.wait(1)
        release.set()
        callback.join(1)
        for thread in spawned:
            thread.join(2)
    idle_add.assert_called_once_with(tray._finish_enabled_converge, False, True, "")
    assert not manager.running


def test_disable_then_enable_converges_to_latest_enabled_intent():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    tray = _bare_tray(manager)
    stopping = threading.Event()
    release_stop = threading.Event()
    calls = []

    def stop():
        calls.append("stop")
        stopping.set()
        assert release_stop.wait(2)
        manager._running = False
        return True, "stopped"

    def start():
        calls.append("start")
        manager._running = True
        return True, "started"

    disabled = Mock()
    disabled.get_active.return_value = False
    enabled = Mock()
    enabled.get_active.return_value = True
    with (
        patch.object(manager, "stop", side_effect=stop),
        patch.object(manager, "start", side_effect=start),
        patch.object(clearvoice, "save_config"),
        patch.object(clearvoice.GLib, "idle_add") as idle_add,
    ):
        tray._on_enable(disabled)
        assert stopping.wait(1)
        worker = tray._enable_converge_thread
        tray._on_enable(enabled)
        assert tray._enable_converge_thread is worker
        release_stop.set()
        worker.join(2)

    assert not worker.is_alive()
    assert calls == ["stop", "start"]
    assert manager.running and tray.config["enabled"]
    idle_add.assert_called_once_with(tray._finish_enabled_converge, True, True, "")


def test_restart_disabled_midway_defers_final_state_to_enable_converger():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    tray = _bare_tray(manager)
    calls = []
    spawned = []
    real_thread = threading.Thread

    def worker_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        spawned.append(thread)
        return thread

    def restart():
        calls.append("restart")
        tray.config["enabled"] = False  # user disables while restart runs
        return True, "restarted"

    def stop():
        calls.append("stop")
        tray.config["enabled"] = True  # user re-enables while stop is pending
        manager._running = False
        return True, "stopped"

    def start():
        calls.append("start")
        manager._running = True
        return True, "started"

    with (
        patch.object(clearvoice.threading, "Thread", side_effect=worker_thread),
        patch.object(manager, "restart", side_effect=restart),
        patch.object(manager, "stop", side_effect=stop),
        patch.object(manager, "start", side_effect=start),
        patch.object(clearvoice.GLib, "idle_add"),
    ):
        tray._async_restart()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and (
            len(spawned) < 2 or any(thread.is_alive() for thread in spawned)
        ):
            time.sleep(0.01)

    assert calls == ["restart", "stop", "start"]
    assert manager.running and tray.config["enabled"]


def test_deferred_launch_replays_mic_demand_when_enable_converges():
    manager = clearvoice.PipelineManager(_config())
    tray = _bare_tray(manager)
    tray._pw_monitor = Mock(nodes_active=True)

    def start():
        manager._running = True
        return True, "started"

    with (
        patch.object(manager, "start", side_effect=start),
        patch.object(manager, "set_mic_demand") as demand,
        patch.object(clearvoice.GLib, "idle_add"),
    ):
        tray._request_enabled_converge(replay_mic=True)
        worker = tray._enable_converge_thread
        worker.join(2)
    assert not worker.is_alive()
    demand.assert_called_once_with(True)


def test_disable_during_deferred_launch_start_stops_and_skips_stale_demand():
    manager = clearvoice.PipelineManager(_config())
    tray = _bare_tray(manager)
    tray._pw_monitor = Mock(nodes_active=True)
    starting = threading.Event()
    release_start = threading.Event()
    calls = []

    def start():
        calls.append("start")
        starting.set()
        assert release_start.wait(2)
        manager._running = True
        return True, "started"

    def stop():
        calls.append("stop")
        manager._running = False
        return True, "stopped"

    disabled = Mock()
    disabled.get_active.return_value = False
    with (
        patch.object(manager, "start", side_effect=start),
        patch.object(manager, "stop", side_effect=stop),
        patch.object(manager, "set_mic_demand") as demand,
        patch.object(clearvoice, "save_config"),
        patch.object(clearvoice.GLib, "idle_add") as idle_add,
    ):
        tray._request_enabled_converge(replay_mic=True)
        assert starting.wait(1)
        worker = tray._enable_converge_thread
        tray._on_enable(disabled)
        release_start.set()
        worker.join(2)

    assert not worker.is_alive()
    assert calls == ["start", "stop"]
    assert not manager.running and not tray.config["enabled"]
    demand.assert_not_called()
    idle_add.assert_called_once_with(tray._finish_enabled_converge, False, True, "")


def test_quit_cleanup_runs_off_gtk_and_suppresses_late_pipeline_start():
    manager = clearvoice.PipelineManager(_config())
    tray = _bare_tray(manager)
    monitor = Mock()
    tray._pw_monitor = monitor
    locked = threading.Event()
    release = threading.Event()
    real_thread = threading.Thread
    spawned = []

    def hold_lifecycle_lock():
        manager._lock.acquire()
        locked.set()
        release.wait(2)
        manager._lock.release()

    def worker_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        spawned.append(thread)
        return thread

    holder = real_thread(target=hold_lifecycle_lock)
    holder.start()
    assert locked.wait(1)
    with (
        patch.object(clearvoice.threading, "Thread", side_effect=worker_thread),
        patch.object(clearvoice, "save_config"),
        patch.object(clearvoice.GLib, "idle_add") as idle_add,
        patch.object(clearvoice.Gtk, "main_quit") as main_quit,
        patch.object(clearvoice.subprocess, "Popen") as popen,
    ):
        tray._on_quit(None)
        assert tray._quitting and manager._shutdown_pending()
        assert manager.start() == (False, "Pipeline shutdown requested")
        assert manager.restart() == (False, "Pipeline shutdown requested")
        popen.assert_not_called()
        main_quit.assert_not_called()
        release.set()
        holder.join(1)
        spawned[0].join(2)
    monitor.stop.assert_called_once()
    idle_add.assert_called_once_with(main_quit)
    main_quit.assert_not_called()


def test_opposing_mic_demand_updates_coalesce_without_blocking_on_link():
    config = _config()
    config["noise_cancellation"]["enabled"] = False
    config["beamforming"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    manager._running = True
    manager._base_mic_node = "mic"
    manager._mic_demand = False
    entered = threading.Event()
    resume = threading.Event()
    links = []

    def link(output_port, input_port, connect):
        links.append((output_port, input_port, connect))
        if connect:
            entered.set()
            assert resume.wait(2)
        return True

    with (
        patch.object(clearvoice, "pw_dump_objects", return_value=_managed_link_graph()),
        patch.object(clearvoice, "pw_link_ports", side_effect=link),
    ):
        assert manager.set_mic_demand(True)
        assert entered.wait(2)
        assert manager.set_mic_demand(False)
        resume.set()
        manager._reconcile_thread.join(3)

    assert not manager._reconcile_thread.is_alive()
    assert not manager._mic_demand
    assert links[0] == ("mic:capture_FL", "clearvoice_ec_capture:input_FL", True)
    assert ("mic:capture_FL", "clearvoice_ec_capture:input_FL", False) in links


def test_output_volume_lock_forces_unity_and_can_be_disabled():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    with (
        patch.object(manager, "_publish_output_volume_lock", return_value=True),
        patch.object(manager, "set_output_gain", return_value=True) as set_gain,
    ):
        assert manager.set_output_volume_lock(True)
        set_gain.assert_called_once_with(100)
        set_gain.reset_mock()
        assert manager.set_output_volume_lock(False)
        set_gain.assert_not_called()


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

    def select_headphone(sink, _gain):
        headphone._playback_sink = sink

    with patch.object(
        headphone, "_set_playback_sink", side_effect=select_headphone
    ) as set_sink:
        headphone._start_playback_output()
    set_sink.assert_called_once_with(physical_sink, 100)

    speaker = clearvoice.PipelineManager(_config())
    speaker._physical_sink = physical_sink
    fake_proc = Mock()

    def select_speaker(sink, _gain):
        speaker._playback_sink = sink

    with tempfile.TemporaryDirectory() as runtime:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(clearvoice.subprocess, "Popen", return_value=fake_proc),
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "pw_wait_for_node", return_value=True),
            patch.object(
                speaker, "_set_playback_sink", side_effect=select_speaker
            ) as set_sink,
        ):
            speaker._start_playback_output()
    assert set_sink.call_args_list[-1].args == (clearvoice.SPEAKER_SINK_NAME, 80)


def _active_beamforming_manager():
    config = _config()
    config["noise_cancellation"]["enabled"] = False
    config["beamforming"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    manager._running = True
    manager._base_mic_node = "mic"
    manager._physical_sink = "alsa_output.physical"
    manager._playback_sink = manager._physical_sink
    manager._mic_demand = True
    manager._demand_revision = 1
    manager._generation = 4
    return manager


def test_aec_link_reconciler_retries_failure_and_repairs_later_loss():
    manager = _active_beamforming_manager()
    graph = _managed_link_graph()
    with (
        patch.object(
            clearvoice,
            "pw_dump_objects",
            side_effect=[graph, graph, _managed_link_graph(("FL",))],
        ),
        patch.object(
            clearvoice,
            "pw_link_ports",
            side_effect=[True, False, True, True, True, True],
        ) as link,
    ):
        manager._reconcile_aec_links(True, 1, 4)
        manager._reconcile_aec_links(True, 1, 4)
        manager._reconcile_aec_links(True, 1, 4)

    calls = [call.args for call in link.call_args_list]
    assert calls == [
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", True),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", True),
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", False),
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", True),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", True),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", True),
    ]


def test_aec_link_unknown_graph_does_not_disconnect_or_clear_state():
    manager = _active_beamforming_manager()
    with (
        patch.object(clearvoice, "pw_dump_objects", return_value=None),
        patch.object(clearvoice, "pw_link_ports") as link,
    ):
        manager._reconcile_aec_links(True, 1, 4)
    link.assert_not_called()
    with patch.object(
        clearvoice.subprocess, "run", return_value=Mock(returncode=1, stderr="offline")
    ):
        assert clearvoice.pw_dump_objects(manager=True) is None


def test_active_reconciliation_repairs_lost_beamformer_to_filter_link():
    config = _config()
    config["beamforming"]["enabled"] = True
    config["speaker_enhancement"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    manager._running = True
    manager._mic_demand = True
    manager._demand_revision = 1
    manager._generation = 6
    manager._base_mic_node = "mic"
    manager._playback_sink = clearvoice.SPEAKER_SINK_NAME
    manager._fc_proc = Mock(pid=51, poll=Mock(return_value=None))
    manager._ec_proc = Mock(pid=52, poll=Mock(return_value=None))
    manager._spk_proc = Mock(pid=53, poll=Mock(return_value=None))

    with (
        patch.object(
            clearvoice,
            "pw_dump_objects",
            return_value=_managed_link_graph(
                linked=("FL", "FR"), include_filter=True
            ),
        ),
        patch.object(clearvoice, "pw_link_ports", return_value=True) as link,
    ):
        manager._reconcile_aec_links(True, 1, 6)

    link.assert_called_once_with(
        "clearvoice_beamformed:capture_MONO",
        "clearvoice_capture:input_MONO",
        True,
    )
    assert manager._fc_proc.poll() is None
    assert manager._ec_proc.poll() is None
    assert manager._spk_proc.poll() is None


def test_filter_link_pairs_only_stereo_front_left_to_mono_capture():
    config = _config()
    manager = clearvoice.PipelineManager(config)
    manager._running = True
    manager._mic_demand = True
    manager._demand_revision = 1
    manager._base_mic_node = "alsa_input.stereo"
    graph = [
        _node("alsa_input.stereo", 1),
        _node("clearvoice_capture", 2),
        {
            "id": 10,
            "type": "PipeWire:Interface:Port",
            "info": {
                "direction": "out",
                "props": {"node.id": 1, "port.name": "capture_FL", "port.direction": "out"},
            },
        },
        {
            "id": 11,
            "type": "PipeWire:Interface:Port",
            "info": {
                "direction": "out",
                "props": {"node.id": 1, "port.name": "capture_FR", "port.direction": "out"},
            },
        },
        {
            "id": 20,
            "type": "PipeWire:Interface:Port",
            "info": {
                "direction": "in",
                "props": {"node.id": 2, "port.name": "input_MONO", "port.direction": "in"},
            },
        },
    ]
    with (
        patch.object(clearvoice, "pw_dump_objects", return_value=graph),
        patch.object(clearvoice, "pw_link_ports") as link,
        patch.object(clearvoice.log, "warning") as warning,
    ):
        manager._reconcile_aec_links(True, 1, 0)
    link.assert_called_once_with(
        "alsa_input.stereo:capture_FL",
        "clearvoice_capture:input_MONO",
        True,
    )
    warning.assert_not_called()


def test_standby_disconnect_failure_is_retried_from_periodic_reconcile_backstop():
    manager = _active_beamforming_manager()
    tray = _bare_tray(manager)
    manager._mic_demand = False
    manager._demand_revision = 2
    manager._ec_proc = Mock(pid=41, poll=Mock(return_value=None))
    graph = _managed_link_graph(("FL", "FR"))
    graph_with_fl = _managed_link_graph(("FL",))
    with (
        patch.object(
            clearvoice,
            "pw_dump_objects",
            side_effect=[graph, graph_with_fl],
        ),
        patch.object(clearvoice, "pw_link_ports", side_effect=[False, True, True]) as link,
        patch.object(manager, "_request_mic_reconcile", wraps=manager._request_mic_reconcile),
    ):
        assert tray._on_reconcile_tick()
        manager._reconcile_thread.join(2)
        assert tray._on_reconcile_tick()
        manager._reconcile_thread.join(2)
    assert [call.args + (call.kwargs["connect"],) for call in link.call_args_list] == [
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", False),
        ("mic:capture_FR", "clearvoice_ec_capture:input_FR", False),
        ("mic:capture_FL", "clearvoice_ec_capture:input_FL", False),
    ]


def test_pw_link_reports_failed_disconnect():
    with patch.object(
        clearvoice.subprocess,
        "run",
        return_value=Mock(returncode=1, stderr="link vanished"),
    ):
        assert not clearvoice.pw_link_ports("source:capture_FL", "target:input_FL", False)


def test_aec_reference_uses_physical_sink_when_speaker_chain_falls_back():
    manager = clearvoice.PipelineManager(_config())
    manager._base_mic_node = "mic"
    manager._physical_sink = "alsa_output.physical"
    with tempfile.TemporaryDirectory() as runtime:
        Path(runtime, "speaker-chain.log").write_text("prior crash output\n")
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(
                clearvoice, "SPEAKER_CHAIN_CONF", Path(runtime, "speaker.conf")
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(clearvoice.subprocess, "Popen", return_value=Mock()) as popen,
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "pw_find_node_id", return_value=8),
            patch.object(
                clearvoice, "pw_set_default_sink", return_value=True
            ) as set_default,
            patch.object(clearvoice, "pw_move_playback_streams"),
            patch.object(clearvoice, "pw_wait_for_node", return_value=False),
        ):
            manager._start_playback_output()
        assert popen.call_args.kwargs["stderr"].closed
        assert "prior crash output" in Path(runtime, "speaker-chain.log").read_text()
        assert "speaker-chain start" in Path(runtime, "speaker-chain.log").read_text()

    assert manager._playback_sink == "alsa_output.physical"
    set_default.assert_called_once_with(8)
    manager.config["echo_cancellation"]["enabled"] = True
    assert (
        "alsa_output.physical:monitor_FL",
        "clearvoice_ec_sink:input_FL",
    ) in manager._aec_link_pairs()


def test_speaker_default_failure_falls_back_to_physical_and_stops_child():
    manager = clearvoice.PipelineManager(_config())
    manager._base_mic_node = "mic"
    manager._physical_sink = "alsa_output.physical"
    speaker_proc = Mock(pid=77, poll=Mock(return_value=None))
    with tempfile.TemporaryDirectory() as runtime:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(
                clearvoice, "SPEAKER_CHAIN_CONF", Path(runtime, "speaker.conf")
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(clearvoice.subprocess, "Popen", return_value=speaker_proc),
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "pw_wait_for_node", return_value=True),
            patch.object(clearvoice, "pw_find_node_id", side_effect=[9, 8]),
            patch.object(clearvoice, "pw_set_default_sink", side_effect=[False, True]),
            patch.object(clearvoice, "pw_move_playback_streams"),
        ):
            manager._start_playback_output()

    assert manager._playback_sink == "alsa_output.physical"
    assert manager._spk_proc is None
    speaker_proc.terminate.assert_called_once()
    manager.config["echo_cancellation"]["enabled"] = True
    assert (
        "alsa_output.physical:monitor_FL",
        "clearvoice_ec_sink:input_FL",
    ) in manager._aec_link_pairs()


def test_missing_playback_selection_reports_unavailable_aec_reference():
    config = _config()
    config["echo_cancellation"]["enabled"] = True
    manager = clearvoice.PipelineManager(config)
    manager._base_mic_node = "mic"
    with patch.object(clearvoice.log, "warning") as warning:
        pairs = manager._aec_link_pairs()
    assert all("clearvoice_ec_sink" not in target for _source, target in pairs)
    warning.assert_called_once()


def test_dead_failed_speaker_child_does_not_trigger_pipeline_health_restart():
    manager = clearvoice.PipelineManager(_config())
    manager._physical_sink = "alsa_output.physical"
    failed_speaker = Mock(pid=77, poll=Mock(return_value=1))
    with tempfile.TemporaryDirectory() as runtime:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(
                clearvoice, "SPEAKER_CHAIN_CONF", Path(runtime, "speaker.conf")
            ),
            patch.object(Path, "is_file", return_value=True),
            patch.object(clearvoice.subprocess, "Popen", return_value=failed_speaker),
            patch.object(clearvoice, "pw_set_node_volume", return_value=True),
            patch.object(clearvoice, "pw_wait_for_node", return_value=False),
            patch.object(clearvoice, "pw_find_node_id", return_value=8),
            patch.object(clearvoice, "pw_set_default_sink", return_value=True),
            patch.object(clearvoice, "pw_move_playback_streams"),
        ):
            manager._start_playback_output()

    mic_process = Mock(pid=41, poll=Mock(return_value=None))
    manager._fc_proc = mic_process
    manager._running = True
    with patch.object(manager, "_request_mic_reconcile"):
        assert manager.check_health()
    assert manager._spk_proc is None
    assert manager._fc_proc is mic_process


def test_health_failure_logs_retained_pid_signal_and_recent_stderr():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    manager._fc_proc = Mock(pid=4321, poll=Mock(return_value=-9))
    with tempfile.TemporaryDirectory() as runtime:
        Path(runtime, "filter-chain.log").write_text(
            "older stderr\n--- filter-chain start ---\nlate plugin panic\n"
        )
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(runtime)),
            patch.object(clearvoice.log, "error") as error,
        ):
            assert not manager.check_health()
    message = error.call_args.args[0] % error.call_args.args[1:]
    assert "pid=4321" in message
    assert "SIGKILL" in message
    assert "late plugin panic" in message


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
    assert "session.suspend-timeout-seconds = 0" in filter_conf
    assert "state.restore-props = false" in filter_conf
    assert "monitor.mode = true" in echo_conf
    assert "state.restore-props = false" in echo_conf
    assert "state.restore-props = false" not in intermediate_echo_conf
    for conf in (filter_conf, echo_conf):
        assert "rt.time.soft = 150000" in conf
        assert "rt.time.hard = 200000" in conf
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


def test_noise_model_config_generation_and_stock_remains_unchanged():
    # sha256 of the stock config generated by the pre-plugin release (commit 9a8078d).
    pinned = {
        (): "5dcd316d9c9a1bb2b9cc6c99165907d8c07d0bd49f40dae8f6848cd8d72e0c40",
        (("studio_voice", False), ("target_source", "mic")): (
            "865a03a3619579282bd2cdbb2dea980bfa7ae953915159aa3c1c828ad717a6dc"
        ),
    }
    for kwargs, digest in pinned.items():
        generated = clearvoice._pw_conf_filter_chain(
            "/tmp/deepfilter.so", model="stock", latency_ms=35, **dict(kwargs)
        )
        assert hashlib.sha256(generated.encode()).hexdigest() == digest
    stock = clearvoice._pw_conf_filter_chain("/tmp/deepfilter.so")
    assert "label  = deep_filter_mono" in stock
    assert '"Latency (ms)"' not in stock
    assert "node.latency = 256/48000" not in stock

    dfn3 = clearvoice._pw_conf_filter_chain(
        "/tmp/clearvoice.so", model="dfn3-ll", latency_ms=35
    )
    assert "label  = clearvoice_dfn3_ll_mono" in dfn3
    assert (
        '"Post Filter Beta" = 0.0\n'
        '                            "Latency (ms)" = 35'
    ) in dfn3
    assert dfn3.count("node.latency = 256/48000") == 2

    fastenhancer = clearvoice._pw_conf_filter_chain(
        "/tmp/clearvoice.so", model="fastenhancer-b", latency_ms=35
    )
    assert "label  = clearvoice_fastenhancer_b_mono" in fastenhancer
    assert (
        'control = {\n                            "Latency (ms)" = 35\n'
        "                        }"
    ) in fastenhancer
    for stock_control in (
        "Attenuation Limit (dB)",
        "Min processing threshold (dB)",
        "Max ERB processing threshold (dB)",
        "Max DF processing threshold (dB)",
        "Post Filter Beta",
    ):
        assert stock_control not in fastenhancer

    fastenhancer_s = clearvoice._pw_conf_filter_chain(
        "/tmp/clearvoice.so", model="fastenhancer-s", latency_ms=80
    )
    assert "label  = clearvoice_fastenhancer_s_mono" in fastenhancer_s
    assert '"Latency (ms)" = 80' in fastenhancer_s


def test_noise_model_config_validation_defaults_unknown_and_clamps_latency():
    assert _config()["noise_cancellation"]["model"] == "stock"
    assert _config()["noise_cancellation"]["latency_ms"] == 35
    assert clearvoice._clamp_latency_ms(1) == 10
    assert clearvoice._clamp_latency_ms(250) == 200
    assert clearvoice._clamp_latency_ms("bad") == 35

    with tempfile.TemporaryDirectory() as directory:
        config_path = Path(directory) / "config.json"
        config_path.write_text(
            json.dumps(
                {"noise_cancellation": {"model": "mystery", "latency_ms": 300}}
            )
        )
        with patch.object(clearvoice, "CONFIG_FILE", config_path), patch.object(
            clearvoice.log, "warning"
        ) as warning:
            loaded = clearvoice.load_config()
    assert loaded["noise_cancellation"]["model"] == "stock"
    assert loaded["noise_cancellation"]["latency_ms"] == 200
    warning.assert_called_once()


def test_missing_optional_plugin_falls_back_without_persisting_preference():
    config = _config()
    config["noise_cancellation"]["model"] = "dfn3-ll"
    manager = clearvoice.PipelineManager(config)
    with tempfile.TemporaryDirectory() as directory:
        missing_plugin = Path(directory) / "missing.so"
        with (
            patch.object(clearvoice, "CLEARVOICE_LADSPA_PLUGIN", missing_plugin),
            patch.object(
                manager, "_launch_filter_chain", return_value=(True, "")
            ) as launch,
            patch.object(clearvoice, "save_config") as save,
        ):
            assert manager._start_filter_chain("mic", "/stock.so")[0]
    assert launch.call_args.args[0:2] == ("stock", "/stock.so")
    assert manager.plugin_fallback_active
    assert manager.noise_model == "dfn3-ll"
    assert config["noise_cancellation"]["model"] == "dfn3-ll"
    assert "not found" in manager.take_fallback_notice()
    save.assert_not_called()


def test_custom_filter_chain_start_failure_retries_stock_once():
    config = _config()
    config["noise_cancellation"]["model"] = "fastenhancer-b"
    manager = clearvoice.PipelineManager(config)
    with tempfile.TemporaryDirectory() as directory:
        plugin = Path(directory) / "plugin.so"
        plugin.touch()
        with (
            patch.object(clearvoice, "CLEARVOICE_LADSPA_PLUGIN", plugin),
            patch.object(
                manager,
                "_launch_filter_chain",
                side_effect=[(False, "no filter node"), (True, "")],
            ) as launch,
            patch.object(clearvoice, "save_config") as save,
        ):
            assert manager._start_filter_chain("mic", "/stock.so")[0]
    assert [call.args[0] for call in launch.call_args_list] == [
        "fastenhancer-b",
        "stock",
    ]
    assert manager.plugin_fallback_active
    assert config["noise_cancellation"]["model"] == "fastenhancer-b"
    save.assert_not_called()


def test_filter_chain_process_exit_is_a_start_failure():
    manager = clearvoice.PipelineManager(_config())
    process = Mock(poll=Mock(return_value=1))
    with patch.object(clearvoice, "pw_node_exists") as node_exists:
        assert not manager._wait_for_filter_chain_node(process)
    node_exists.assert_not_called()


def test_noise_model_menu_saves_preference_and_restarts_running_pipeline():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    tray = _bare_tray(manager)
    tray.config = manager.config
    tray._async_restart = Mock()
    item = Mock()
    item.get_active.return_value = True
    with patch.object(clearvoice, "save_config") as save:
        tray._on_noise_model(item, "dfn3-ll")
    assert manager.config["noise_cancellation"]["model"] == "dfn3-ll"
    save.assert_called_once_with(manager.config)
    tray._async_restart.assert_called_once_with()

    manager._plugin_fallback = True
    manager._fc_model = "stock"
    tray._mi_status = Mock()
    clearvoice.ClearVoiceTray._update_status(tray)
    status = tray._mi_status.set_label.call_args.args[0]
    assert "Stock DeepFilterNet (fallback)" in status


def _launch_running_filter_chain(manager, runtime: Path, model: str):
    """Run the real launcher with only the child process and node wait mocked."""
    child = Mock(pid=51, poll=Mock(return_value=None))
    with (
        patch.object(clearvoice, "RUNTIME_DIR", runtime),
        patch.object(manager, "_spawn_child", return_value=child),
        patch.object(manager, "_wait_for_filter_chain_node", return_value=True),
    ):
        ok, _ = manager._launch_filter_chain(model, "/tmp/plugin.so", "mic", {})
    assert ok
    manager._running = True
    return runtime / "filter-chain.log"


FATAL_LINE = "clearvoice-ladspa fatal label=clearvoice_dfn3_ll_mono reason=backend\n"


def test_fatal_plugin_log_after_node_exists_triggers_health_restart_fallback():
    config = _config()
    config["noise_cancellation"]["model"] = "dfn3-ll"
    manager = clearvoice.PipelineManager(config)
    with tempfile.TemporaryDirectory() as directory:
        runtime = Path(directory)
        # A fatal line from an earlier attempt must never count for this one.
        (runtime / "filter-chain.log").write_text(FATAL_LINE)
        log_path = _launch_running_filter_chain(manager, runtime, "dfn3-ll")
        with patch.object(clearvoice, "save_config") as save:
            assert manager.check_health()
            assert not manager.plugin_fallback_active
            # The child writes a fatal line after its node already exists.
            with open(log_path, "a") as child_stderr:
                child_stderr.write(FATAL_LINE)
            assert not manager.check_health()
        manager._close_filter_chain_log()
    assert manager.plugin_fallback_active
    assert config["noise_cancellation"]["model"] == "dfn3-ll"
    assert "fatal" in manager.take_fallback_notice()
    save.assert_not_called()


def test_fatal_line_is_read_from_child_stderr_inode_after_log_path_rotation():
    manager = clearvoice.PipelineManager(_config())
    with tempfile.TemporaryDirectory() as directory:
        runtime = Path(directory)
        log_path = _launch_running_filter_chain(manager, runtime, "dfn3-ll")
        rotated = runtime / "filter-chain.log.1"
        log_path.rename(rotated)
        log_path.write_text("")
        with open(rotated, "a") as child_stderr:  # child keeps its original inode
            child_stderr.write(FATAL_LINE)
        assert not manager.check_health()
        manager._close_filter_chain_log()
    assert manager.plugin_fallback_active


def test_fatal_line_is_handled_while_promotion_waits_for_missing_worker():
    config = _config()
    config["noise_cancellation"]["model"] = "dfn3-ll"
    manager = clearvoice.PipelineManager(config)
    waiting = threading.Event()
    release = threading.Event()

    def promotion_waiting_for_worker(*_args, **_kwargs):
        waiting.set()
        release.wait(5)

    with tempfile.TemporaryDirectory() as directory:
        log_path = _launch_running_filter_chain(manager, Path(directory), "dfn3-ll")
        manager._mic_demand = True
        manager._reconcile_requested = True
        manager._reconcile_running = True
        with (
            patch.object(manager, "_reconcile_aec_links"),
            patch.object(
                clearvoice,
                "promote_deepfilter_worker",
                side_effect=promotion_waiting_for_worker,
            ),
        ):
            reconciler = threading.Thread(target=manager._mic_reconcile_loop)
            reconciler.start()
            try:
                assert waiting.wait(5)
                with open(log_path, "a") as child_stderr:
                    child_stderr.write(FATAL_LINE)
                assert not manager.check_health()
            finally:
                release.set()
                reconciler.join(5)
        manager._close_filter_chain_log()
    assert manager.plugin_fallback_active


def test_health_check_skips_while_lifecycle_lock_is_held():
    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    manager._fc_model = "fastenhancer-b"
    manager._fc_proc = Mock(pid=61, poll=Mock(return_value=-15))  # intentional SIGTERM
    with manager._lock:
        assert manager.check_health()
    assert manager._fc_crash_times == []
    assert not manager.plugin_fallback_active


def test_two_custom_filter_chain_crashes_in_60_seconds_fall_back_to_stock():
    config = _config()
    config["noise_cancellation"]["model"] = "fastenhancer-s"
    manager = clearvoice.PipelineManager(config)
    manager._running = True
    manager._fc_model = "fastenhancer-s"
    manager._fc_proc = Mock(pid=81, poll=Mock(return_value=1))
    clock = [10.0]
    with tempfile.TemporaryDirectory() as directory:
        with (
            patch.object(clearvoice, "RUNTIME_DIR", Path(directory)),
            patch.object(
                clearvoice.time, "monotonic", side_effect=lambda: clock[0]
            ),
            patch.object(clearvoice, "save_config") as save,
        ):
            assert not manager.check_health()
            assert not manager.plugin_fallback_active
            manager._fc_crash_counted = False
            manager._fc_proc = Mock(pid=82, poll=Mock(return_value=-9))
            clock[0] = 50.0
            assert not manager.check_health()
    assert manager.plugin_fallback_active
    assert "twice within 60 seconds" in manager.take_fallback_notice()
    assert config["noise_cancellation"]["model"] == "fastenhancer-s"
    save.assert_not_called()


def test_clearvoice_stats_warn_only_above_five_percent_concealment():
    manager = clearvoice.PipelineManager(_config())
    assert clearvoice._parse_clearvoice_stats(
        "clearvoice-ladspa stats label=clearvoice_dfn3_ll_mono "
        "processed=100 concealed=7 discarded_late=0 input_dropped=0 "
        "output_dropped=0 settling=0 throttled=0 oversized_blocks=0"
    ) == {"label": "clearvoice_dfn3_ll_mono", "processed": 100, "concealed": 7}

    def stats(processed, concealed):
        return (
            "clearvoice-ladspa stats label=clearvoice_dfn3_ll_mono "
            f"processed={processed} concealed={concealed}\n"
        )

    clock = [100.0]
    with tempfile.TemporaryDirectory() as directory:
        log_path = _launch_running_filter_chain(manager, Path(directory), "dfn3-ll")
        with (
            patch.object(clearvoice.time, "monotonic", side_effect=lambda: clock[0]),
            patch.object(clearvoice.log, "warning") as warning,
        ):

            def tick(*lines):
                with open(log_path, "a") as child_stderr:
                    child_stderr.writelines(lines)
                assert manager.check_health()

            tick(stats(100, 0), stats(200, 5))  # exactly 5%: no warning
            assert warning.call_count == 0
            tick(stats(300, 11))  # 6%: warns
            assert warning.call_count == 1
            clock[0] = 130.0
            tick(stats(400, 30))  # 19% within 60 s: rate-limited
            assert warning.call_count == 1
            tick(stats(50, 40))  # new plugin instance: counters reset, no delta
            assert warning.call_count == 1
            clock[0] = 161.0
            tick(stats(150, 50))  # 10% after 60 s: warns again
            assert warning.call_count == 2
        manager._close_filter_chain_log()
    assert not manager.plugin_fallback_active


def test_worker_selection_uses_exact_custom_name_and_keeps_stock_layout():
    ps_result = Mock(
        returncode=0,
        stdout=(
            "100 pipewire TS\n101 unrelated-worker TS\n"
            "102 cv-dsp-worker TS\n103 data-loop.0 RR\n"
        ),
    )
    with patch.object(clearvoice.subprocess, "run", return_value=ps_result):
        assert clearvoice._deepfilter_thread(100, worker_name="cv-dsp-worker") == (
            102,
            "",
        )
        stock_worker, reason = clearvoice._deepfilter_thread(100)
    assert stock_worker is None
    assert reason.startswith("ambiguous")

    manager = clearvoice.PipelineManager(_config())
    manager._running = True
    manager._mic_demand = True
    manager._fc_model = "fastenhancer-b"
    manager._fc_proc = Mock(pid=101, poll=Mock(return_value=None))
    with patch.object(clearvoice, "promote_deepfilter_worker") as promote:
        manager._reconcile_deepfilter(True, 0, 0)
    assert promote.call_args.kwargs["worker_name"] == "cv-dsp-worker"


def main():
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
    print("ClearVoice tests passed")


if __name__ == "__main__":
    main()
