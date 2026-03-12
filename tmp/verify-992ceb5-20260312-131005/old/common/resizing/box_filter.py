from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cupy as cp
import numpy as np

from common.kernel_loader import cuda_compile_options, read_cuda_source

from .base import CupyRgbImageResizer

_BLOCK = (16, 16, 1)
_CUDA_DIR = Path(__file__).with_name("cuda")


@lru_cache(maxsize=1)
def _box_filter_kernel() -> cp.RawKernel:
    source = read_cuda_source(_CUDA_DIR, "box_filter.cu")
    return cp.RawKernel(
        source,
        "box_filter_resize_kernel",
        options=cuda_compile_options(_CUDA_DIR, "-std=c++11"),
    )


class BoxFilterResizer(CupyRgbImageResizer):
    def __init__(self) -> None:
        self.kernel = _box_filter_kernel()

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
        return output
