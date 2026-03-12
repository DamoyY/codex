from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from common.kernel_loader import cuda_compile_options, read_cuda_source

if TYPE_CHECKING:
    from .plans import AxisPlan

_BLOCK = (16, 8, 1)
_CUDA_DIR = Path(__file__).with_name("cuda")


@lru_cache(maxsize=1)
def _get_kernels() -> tuple[cp.RawKernel, cp.RawKernel]:
    source = read_cuda_source(_CUDA_DIR, "separable.cu")
    options = cuda_compile_options(_CUDA_DIR, "-std=c++11")
    return (
        cp.RawKernel(source, "horizontal_resize_kernel", options=options),
        cp.RawKernel(source, "vertical_resize_kernel", options=options),
    )


def _grid(
    width: int,
    height: int,
    batch_size: int,
) -> tuple[int, int, int]:
    return (
        (width + _BLOCK[0] - 1) // _BLOCK[0],
        (height + _BLOCK[1] - 1) // _BLOCK[1],
        batch_size,
    )


def _validate_output_array(
    output: cp.ndarray,
    expected_shape: tuple[int, ...],
    name: str,
) -> cp.ndarray:
    if (
        output.shape != expected_shape
        or output.dtype != cp.float32
        or not output.flags.c_contiguous
    ):
        msg = f"{name} 输出数组不符合要求。"
        raise ValueError(msg)
    return output


def launch_horizontal(
    image_array: cp.ndarray,
    plan: AxisPlan,
    target_width: int,
    *,
    clamp_output: bool,
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    kernel, _ = _get_kernels()
    source_batch, source_height, source_width, _ = image_array.shape
    expected_shape = (source_batch, source_height, target_width, 3)
    if output is None:
        output = cp.empty(expected_shape, dtype=cp.float32)
    else:
        output = _validate_output_array(
            output,
            expected_shape,
            "horizontal",
        )
    kernel(
        _grid(target_width, source_height, source_batch),
        _BLOCK,
        (
            image_array,
            np.int32(source_batch),
            np.int32(source_height),
            np.int32(source_width),
            plan.indices,
            plan.weights,
            np.int32(plan.taps),
            output,
            np.int32(target_width),
            np.int32(int(clamp_output)),
        ),
    )
    return output


def launch_vertical(
    image_array: cp.ndarray,
    plan: AxisPlan,
    target_height: int,
    *,
    clamp_output: bool,
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    _, kernel = _get_kernels()
    source_batch, source_height, source_width, _ = image_array.shape
    expected_shape = (source_batch, target_height, source_width, 3)
    if output is None:
        output = cp.empty(expected_shape, dtype=cp.float32)
    else:
        output = _validate_output_array(
            output,
            expected_shape,
            "vertical",
        )
    kernel(
        _grid(source_width, target_height, source_batch),
        _BLOCK,
        (
            image_array,
            np.int32(source_batch),
            np.int32(source_height),
            np.int32(source_width),
            plan.indices,
            plan.weights,
            np.int32(plan.taps),
            output,
            np.int32(target_height),
            np.int32(int(clamp_output)),
        ),
    )
    return output
