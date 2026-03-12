from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from .base import CupyRgbImageResizer
from .separable_kernels import launch_horizontal, launch_vertical

if TYPE_CHECKING:
    import cupy as cp

    from .plans import AxisPlan


class SeparableResizer(CupyRgbImageResizer):
    def __init__(
        self,
        *,
        clamp_output: bool,
        skip_identity_axes: bool,
        cache_size: int = 512,
    ) -> None:
        self._clamp_output = clamp_output
        self._skip_identity_axes = skip_identity_axes
        self._cached_build_axis_plan = lru_cache(maxsize=cache_size)(
            self._build_axis_plan,
        )

    def _build_axis_plan(self, src_length: int, dst_length: int) -> AxisPlan:
        raise NotImplementedError

    def _get_axis_plan(
        self,
        src_length: int,
        dst_length: int,
    ) -> AxisPlan | None:
        if self._skip_identity_axes and src_length == dst_length:
            return None
        return self._cached_build_axis_plan(src_length, dst_length)

    def _resize_once(
        self,
        image_array: cp.ndarray,
        target_size: tuple[int, int],
        *,
        output: cp.ndarray | None = None,
    ) -> cp.ndarray:
        image_array = self._as_nhwc_batch_array(image_array)
        batch_size, source_height, source_width, _ = image_array.shape
        target_width, target_height = target_size
        x_plan = self._get_axis_plan(source_width, target_width)
        y_plan = self._get_axis_plan(source_height, target_height)
        if x_plan is None and y_plan is None:
            if output is None:
                return image_array.copy()
            resolved_output = self._require_nhwc_output_array(
                output,
                target_size,
                batch_size=batch_size,
            )
            resolved_output[...] = image_array
            return resolved_output
        if x_plan is None:
            if y_plan is None:
                msg = "缺少垂直缩放计划。"
                raise ValueError(msg)
            return launch_vertical(
                image_array,
                y_plan,
                target_height,
                clamp_output=self._clamp_output,
                output=output,
            )
        if y_plan is None:
            return launch_horizontal(
                image_array,
                x_plan,
                target_width,
                clamp_output=self._clamp_output,
                output=output,
            )
        intermediate = launch_horizontal(
            image_array,
            x_plan,
            target_width,
            clamp_output=False,
        )
        return launch_vertical(
            intermediate,
            y_plan,
            target_height,
            clamp_output=self._clamp_output,
            output=output,
        )
