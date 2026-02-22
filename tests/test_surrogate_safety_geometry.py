from __future__ import annotations

import pandas as pd

from cutin_risk.indicators.surrogate_safety import (
    IndicatorOptions,
    LongitudinalModel,
    compute_pair_indicators_at_frame,
)


def _row(*, x: float, x_velocity: float, length: float) -> pd.Series:
    return pd.Series({"x": x, "xVelocity": x_velocity, "width": length})


def test_bbox_topleft_sign_positive_produces_positive_gap() -> None:
    leader = _row(x=100.0, x_velocity=10.0, length=4.0)
    follower = _row(x=90.0, x_velocity=11.0, length=4.0)
    out = compute_pair_indicators_at_frame(
        leader,
        follower,
        leader_sign=1,
        follower_sign=1,
        model=LongitudinalModel(position_reference="bbox_topleft"),
        options=IndicatorOptions(),
    )
    assert out["dhw"] == 6.0


def test_bbox_topleft_sign_negative_handles_highd_top_left_geometry() -> None:
    # drivingDirection=-1 layout where highD x corresponds to front bumper.
    # leader is ahead (smaller x), follower is behind (larger x).
    leader = _row(x=90.0, x_velocity=-30.0, length=4.0)
    follower = _row(x=100.0, x_velocity=-24.0, length=19.0)
    out_bbox = compute_pair_indicators_at_frame(
        leader,
        follower,
        leader_sign=-1,
        follower_sign=-1,
        model=LongitudinalModel(position_reference="bbox_topleft"),
        options=IndicatorOptions(),
    )
    out_rear = compute_pair_indicators_at_frame(
        leader,
        follower,
        leader_sign=-1,
        follower_sign=-1,
        model=LongitudinalModel(position_reference="rear"),
        options=IndicatorOptions(),
    )

    # Correct geometry yields a positive 6 m bumper-to-bumper gap.
    assert out_bbox["dhw"] == 6.0
    # Legacy rear-reference assumption yields the known negative-gap artifact.
    assert out_rear["dhw"] == -9.0
