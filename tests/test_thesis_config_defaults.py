from __future__ import annotations

import json
from pathlib import Path


def test_default_indicator_position_reference_is_bbox_topleft() -> None:
    cfg = Path("configs/thesis.json")
    data = json.loads(cfg.read_text(encoding="utf-8"))
    assert data["indicators"]["position_reference"] == "bbox_topleft"
