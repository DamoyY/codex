from typing import Any

import cupy as cp
import torch
from torch.utils import dlpack


def move_tensor_to_device(
    tensor: torch.Tensor,
    device: torch.device | None,
    *,
    channels_last: bool,
) -> torch.Tensor:
    if device is None:
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        if channels_last and tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        return tensor
    is_cpu_to_cuda = tensor.device.type == "cpu" and device.type == "cuda"
    if (
        tensor.device == device
        and tensor.dtype == torch.float32
        and (
            not channels_last
            or tensor.ndim != 4
            or tensor.is_contiguous(memory_format=torch.channels_last)
        )
    ):
        return tensor
    if is_cpu_to_cuda:
        tensor = tensor.pin_memory()
    memory_format = (
        torch.channels_last
        if channels_last and tensor.ndim == 4
        else torch.contiguous_format
    )
    return tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=is_cpu_to_cuda,
        memory_format=memory_format,
    )


def cupy_images_to_tensor_batch(
    image_arrays: list[object],
    device: torch.device | None = None,
    *,
    channels_last: bool = False,
) -> torch.Tensor:
    if not image_arrays:
        msg = "image_arrays 不能为空。"
        raise ValueError(msg)
    gpu_arrays: list[Any] = []
    for image_array in image_arrays:
        gpu_array = cp.ascontiguousarray(image_array, dtype=cp.float32)
        if gpu_array.ndim != 4 or gpu_array.shape[3] != 3:
            msg = "image_arrays 中的每个数组都必须为 NHWC RGB。"
            raise ValueError(msg)
        if gpu_array.shape[0] == 0:
            msg = "image_arrays 中的批次数组不能为空。"
            raise ValueError(msg)
        gpu_arrays.append(gpu_array)
    batch_array = cp.concatenate(gpu_arrays, axis=0)
    return cupy_batch_array_to_tensor(
        batch_array,
        device=device,
        channels_last=channels_last,
    )


def cupy_batch_array_to_tensor(
    batch_array: object,
    device: torch.device | None = None,
    *,
    channels_last: bool = False,
) -> torch.Tensor:
    batch_array_any: Any = cp.ascontiguousarray(
        batch_array,
        dtype=cp.float32,
    )
    if batch_array_any.ndim != 4 or batch_array_any.shape[3] != 3:
        msg = "batch_array 必须为 NHWC RGB。"
        raise ValueError(msg)
    tensor = dlpack.from_dlpack(batch_array_any)
    tensor = tensor.permute(0, 3, 1, 2)
    return move_tensor_to_device(
        tensor,
        device=device,
        channels_last=channels_last,
    )
