"""Placeholder hooks for optional filtering and normalization of detected events."""

from __future__ import annotations


def postprocess_events(events):
    """Apply event-level cleanup after raw detection, once implemented."""
    raise NotImplementedError("Postprocessing not implemented.")
