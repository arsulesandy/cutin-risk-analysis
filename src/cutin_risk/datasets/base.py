from __future__ import annotations


class DatasetAdapter:
    """Base interface for trajectory datasets."""

    def load(self, source: str):
        """Load dataset contents from the provided source path."""
        raise NotImplementedError("DatasetAdapter.load must be implemented.")
