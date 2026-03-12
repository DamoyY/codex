from __future__ import annotations

from .plans import AxisPlan, build_lanczos3_axis_plan
from .separable import SeparableResizer


class Lanczos3Resizer(SeparableResizer):
    def __init__(self) -> None:
        super().__init__(clamp_output=True, skip_identity_axes=False)

    def _build_axis_plan(self, src_length: int, dst_length: int) -> AxisPlan:
        return build_lanczos3_axis_plan(src_length, dst_length)
