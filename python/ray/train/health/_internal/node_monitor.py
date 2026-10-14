import threading
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Set

from ray.train.health.probe import NodeProbe, ProbeResult


@dataclass
class NodeMonitorStatus:
    """The results of ``NodeProbe``\\ s on one node.

    Attributes:
        probe_results: ``{probe name: ProbeResult}`` for the probes that
            finished since the last poll.
        probe_errors: ``{probe name: exception}`` for the probes that raised
            since the last poll, or that were asked for again while still
            running.
        error: An error reaching the monitor, if any.
    """

    probe_results: Dict[str, ProbeResult] = field(default_factory=dict)
    probe_errors: Dict[str, Exception] = field(default_factory=dict)
    error: Optional[Exception] = None


class NodeMonitor:
    """Runs ``NodeProbe``\\ s on the node it runs on, each in its own thread.

    Args:
        probes: The probes it can run.
    """

    def __init__(self, probes: List[NodeProbe]):
        self._probes = {probe.probe_name(): probe for probe in probes}
        self._lock = threading.Lock()
        self._running: Set[str] = set()
        self._results: Dict[str, ProbeResult] = {}
        self._errors: Dict[str, Exception] = {}

    def poll_status(
        self, probe_names: List[str], timeout_s: float
    ) -> NodeMonitorStatus:
        """Start the given probes, wait up to ``timeout_s`` for them to finish,
        and return the results of the probes that finished since the last poll.

        A probe that is still running from an earlier poll is not started again;
        it is reported as an error instead.

        Args:
            probe_names: The probes to start.
            timeout_s: How long to wait for the probes it starts.

        Returns:
            The results and errors since the last poll.
        """
        threads = []
        with self._lock:
            for name in probe_names:
                if name in self._running:
                    self._errors[name] = TimeoutError(
                        f"Node probe {name} is still running from an earlier poll."
                    )
                    continue
                self._running.add(name)
                threads.append(
                    threading.Thread(
                        target=self._run,
                        args=(name,),
                        name=f"NodeMonitor-{name}",
                        daemon=True,
                    )
                )
        for thread in threads:
            thread.start()

        deadline = time.monotonic() + timeout_s
        for thread in threads:
            thread.join(max(deadline - time.monotonic(), 0))

        with self._lock:
            status = NodeMonitorStatus(
                probe_results=self._results, probe_errors=self._errors
            )
            self._results, self._errors = {}, {}
        return status

    def _run(self, name: str) -> None:
        """Run one probe and keep its result for ``poll_status()``. A result
        without a ``timestamp_s`` is stamped with the time the probe
        returned."""
        try:
            result = self._probes[name].poll()
        except Exception as e:
            with self._lock:
                self._errors[name] = e
                self._running.discard(name)
            return
        if result.timestamp_s is None:
            result = replace(result, timestamp_s=time.time())
        with self._lock:
            self._results[name] = result
            self._running.discard(name)
