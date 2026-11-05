"""A minimal in-process event bus.

Agents publish named events (e.g. ``TAXONOMY_UPDATED``, ``BLADE_INSPECTED``,
``TAXONOMY_DELTA``) and the Orchestrator / other agents subscribe. This keeps
the pipeline loosely coupled per the spec's event-driven architecture.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventBus:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        self._subscribers[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        if event in self._subscribers and handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)

    def publish(self, event: str, payload: Dict[str, Any]) -> None:
        for handler in list(self._subscribers.get(event, [])):
            handler(payload)

    # Convenience constants for well-known events.
    TAXONOMY_UPDATED = "TAXONOMY_UPDATED"
    TAXONOMY_DELTA = "TAXONOMY_DELTA"
    BLADE_INSPECTED = "BLADE_INSPECTED"
    TURBINE_INSPECTED = "TURBINE_INSPECTED"
    REPORT_GENERATED = "REPORT_GENERATED"
    REEXAMINE_REQUESTED = "REEXAMINE_REQUESTED"


# Well-known event names, exported for ergonomics.
TAXONOMY_UPDATED = EventBus.TAXONOMY_UPDATED
TAXONOMY_DELTA = EventBus.TAXONOMY_DELTA
BLADE_INSPECTED = EventBus.BLADE_INSPECTED
TURBINE_INSPECTED = EventBus.TURBINE_INSPECTED
REPORT_GENERATED = EventBus.REPORT_GENERATED
REEXAMINE_REQUESTED = EventBus.REEXAMINE_REQUESTED
