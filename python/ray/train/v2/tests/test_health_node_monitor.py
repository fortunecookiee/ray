import os
import sys
import time

import pytest

import ray
from ray.train.health import NodeProbe, ProbeResult
from ray.train.health._internal.node_monitor import NodeMonitor
from ray.train.health._internal.node_monitor_group import NodeMonitorGroup


class Temp(NodeProbe):
    def poll(self):
        return ProbeResult(metrics={"temp_c": 60.0})


class Pid(NodeProbe):
    def poll(self):
        return ProbeResult(metrics={"pid": float(os.getpid())})


class Broken(NodeProbe):
    def poll(self):
        raise RuntimeError("sensor gone")


class Counter(NodeProbe):
    def __init__(self, seconds=0.0):
        self.seconds = seconds
        self.runs = 0

    def poll(self):
        self.runs += 1
        time.sleep(self.seconds)
        return ProbeResult(metrics={"runs": float(self.runs)})


class Hang(NodeProbe):
    def __init__(self):
        self.runs = 0

    def poll(self):
        self.runs += 1
        time.sleep(600)


def _wait_for(fn, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail("timed out")


def test_poll_status_returns_the_results_of_the_probes_it_starts():
    counter = Counter()
    monitor = NodeMonitor([Temp(), counter])

    results = monitor.poll_status(["Temp"], timeout_s=5).probe_results
    assert results["Temp"].metrics == {"temp_c": 60.0}
    assert results["Temp"].timestamp_s is not None
    assert counter.runs == 0


def test_a_failing_probe_is_reported_and_the_others_still_return():
    monitor = NodeMonitor([Broken(), Temp()])
    status = monitor.poll_status(["Broken", "Temp"], timeout_s=5)
    assert isinstance(status.probe_errors["Broken"], RuntimeError)
    assert status.probe_results["Temp"].metrics == {"temp_c": 60.0}


def test_a_slower_probe_is_returned_by_a_later_poll():
    monitor = NodeMonitor([Counter(seconds=0.5)])
    started = time.monotonic()
    assert monitor.poll_status(["Counter"], timeout_s=0.1).probe_results == {}
    assert time.monotonic() - started < 0.4

    results = _wait_for(lambda: monitor.poll_status([], timeout_s=0).probe_results)
    assert results["Counter"].metrics == {"runs": 1.0}


def test_a_probe_still_running_when_asked_again_is_reported_and_not_restarted():
    hang = Hang()
    monitor = NodeMonitor([hang, Temp()])
    started = time.monotonic()
    first = monitor.poll_status(["Hang", "Temp"], timeout_s=0.3)
    assert time.monotonic() - started < 1.0
    assert first.probe_errors == {}
    assert first.probe_results["Temp"].metrics == {"temp_c": 60.0}

    second = monitor.poll_status(["Hang", "Temp"], timeout_s=0.3)
    assert isinstance(second.probe_errors["Hang"], TimeoutError)
    assert second.probe_results["Temp"].metrics == {"temp_c": 60.0}
    assert hang.runs == 1


def _local_node_id():
    return ray.get_runtime_context().get_node_id()


def test_monitors_run_probes_on_their_node_and_stop_on_shutdown(ray_start_4_cpus):
    node_id = _local_node_id()
    group = NodeMonitorGroup([Temp(), Pid()])
    group.start([node_id])

    results = _wait_for(
        lambda: group.poll_status(["Pid"], timeout=30)
        .node_monitor_statuses[node_id]
        .probe_results,
        timeout_s=30.0,
    )
    assert results["Pid"].metrics["pid"] != os.getpid()
    assert "Temp" not in results

    group.shutdown()
    assert group.poll_status(["Pid"], timeout=1).node_monitor_statuses == {}


def test_start_adds_monitors_only_where_missing_and_shutdown_can_stop_some(
    ray_start_4_cpus,
):
    node_id = _local_node_id()
    group = NodeMonitorGroup([Temp()])
    group.start([node_id])
    monitor = group._monitors[node_id]

    group.start([node_id])
    assert group._monitors[node_id] is monitor

    group.shutdown(["some-other-node"])
    assert group.node_ids == [node_id]
    group.shutdown([node_id])
    assert group.node_ids == []
    with pytest.raises(ray.exceptions.RayActorError):
        ray.get(monitor.poll_status.remote([], 0))


@ray.remote(num_cpus=0)
class UnresponsiveMonitor:
    def poll_status(self, probe_names, timeout_s):
        time.sleep(600)


def test_an_unresponsive_monitor_is_reported_after_the_health_check_timeout(
    ray_start_4_cpus,
):
    group = NodeMonitorGroup([], health_check_timeout_s=3.0)
    group._monitors["n1"] = UnresponsiveMonitor.remote()

    # The default timeout keeps a monitor that never answers from blocking.
    started = time.monotonic()
    first = group.poll_status(["Temp"]).node_monitor_statuses["n1"]
    assert first.error is None
    assert time.monotonic() - started < 3

    time.sleep(2.5)
    second = group.poll_status([], timeout=0.2).node_monitor_statuses["n1"]
    assert isinstance(second.error, TimeoutError)


def test_a_dead_monitor_is_reported(ray_start_4_cpus):
    node_id = _local_node_id()
    group = NodeMonitorGroup([Temp()])
    group.start([node_id])
    group.poll_status(["Temp"], timeout=30)

    ray.kill(group._monitors[node_id])

    # The kill is asynchronous, so a poll right after it may still be answered.
    errors = _wait_for(lambda: group.poll_status([], timeout=30).errors)
    assert isinstance(errors[node_id], ray.exceptions.RayActorError)
    group.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", "-x", __file__]))
