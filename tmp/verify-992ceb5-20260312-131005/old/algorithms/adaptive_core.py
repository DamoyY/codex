from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from common.kernel_loader import cuda_compile_options, read_cuda_source
from common.resizing.base import CupyRgbImageResizer
from common.resizing.box_filter import BoxFilterResizer

_BLOCK = (16, 16, 1)
_CUDA_DIR = Path(__file__).with_name("cuda")


@lru_cache(maxsize=1)
def _adaptive_kernel() -> cp.RawKernel:
    source = read_cuda_source(_CUDA_DIR, "adaptive_upscaling.cu")
    return cp.RawKernel(
        source,
        "adaptive_upscaling_kernel",
        options=cuda_compile_options(_CUDA_DIR, "-std=c++11"),
    )


class AdaptiveUpscaler(CupyRgbImageResizer):
    def __init__(self) -> None:
        self.box_filter = BoxFilterResizer()
        self.kernel = _adaptive_kernel()

    def _build_pass_sizes(
        self,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if source_size == target_size:
            return []
        source_width, source_height = source_size
        target_width, target_height = target_size
        if target_width <= source_width and target_height <= source_height:
            return [target_size]
        pass_sizes: list[tuple[int, int]] = []
        current_width, current_height = source_size
        while (
            target_width > current_width * 1.5
            or target_height > current_height * 1.5
        ):
            next_width = min(
                target_width,
                max(current_width + 1, int(current_width * 1.5)),
            )
            next_height = min(
                target_height,
                max(current_height + 1, int(current_height * 1.5)),
            )
            pass_sizes.append((next_width, next_height))
            current_width, current_height = next_width, next_height
        pass_sizes.append(target_size)
        return pass_sizes

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
        output = (
            cp.empty(
                (batch_size, target_height, target_width, 3),
                dtype=cp.float32,
            )
            if output is None
            else self._require_nhwc_output_array(
                output,
                target_size,
                batch_size=batch_size,
            )
        )
        grid = (
            (target_width + _BLOCK[0] - 1) // _BLOCK[0],
            (target_height + _BLOCK[1] - 1) // _BLOCK[1],
            batch_size,
        )
        self.kernel(
            grid,
            _BLOCK,
            (
                image_array,
                np.int32(batch_size),
                np.int32(source_height),
                np.int32(source_width),
                output,
                np.int32(target_height),
                np.int32(target_width),
            ),
        )
        if target_width < source_width or target_height < source_height:
            downscale = max(
                source_width / target_width,
                source_height / target_height,
            )
            box_mix = min(max((downscale - 10.0) / 6.0, 0.0), 0.75)
            if box_mix > 0.0:
                box_output = self.box_filter.resize_array(
                    image_array,
                    target_size,
                )
                output += (box_output - output) * box_mix
        return output
