from __future__ import annotations
from typing import Dict, Type

from .base import DatasetAdapter

DATASET_REGISTRY: Dict[str, Type[DatasetAdapter]] = {}


def register_dataset(name: str, adapter: Type[DatasetAdapter]) -> None:
    """Register a dataset adapter by name."""
    DATASET_REGISTRY[name] = adapter


def get_dataset(name: str) -> Type[DatasetAdapter] | None:
    """Retrieve a registered dataset adapter."""
    return DATASET_REGISTRY.get(name)
