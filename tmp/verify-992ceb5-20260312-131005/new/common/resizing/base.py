from __future__ import annotations

import math

import cupy as cp
import numpy as np
from PIL import Image


class CupyRgbImageResizer:

    def resize_array(
        self,
        image: Image.Image | np.ndarray | cp.ndarray,
        target_size: tuple[int, int],
        *,
        output: cp.ndarray | None = None,
    ) -> cp.ndarray:
        image_array, source_size = self._prepare_image_array(image)
        batch_size = image_array.shape[0]
        resolved_output = (
            None
            if output is None
            else self._require_nhwc_output_array(
                output,
                target_size,
                batch_size=batch_size,
            )
        )
        if target_size == source_size:
            if resolved_output is None:
                return image_array.copy()
            resolved_output[...] = image_array
            return resolved_output

        pass_sizes = self._build_pass_sizes(source_size, target_size)
        final_pass_index = len(pass_sizes) - 1
        for pass_index, pass_size in enumerate(pass_sizes):
            image_array = self._resize_once(
                image_array,
                pass_size,
                output=(
                    resolved_output if pass_index == final_pass_index else None
                ),
            )
        return image_array

    def _prepare_image_array(
        self,
        image: Image.Image | np.ndarray | cp.ndarray,
    ) -> tuple[cp.ndarray, tuple[int, int]]:
        if isinstance(image, Image.Image):
            if image.mode != "RGB":
                msg = "输入图像必须为 RGB。"
                raise ValueError(msg)
            image_array = cp.asarray(
                np.asarray(image, dtype=np.float32) / 255.0,
                dtype=cp.float32,
            )
            image_array = cp.ascontiguousarray(image_array)
            return cp.expand_dims(image_array, axis=0), image.size
        if isinstance(image, np.ndarray):
            self._validate_nhwc_rgb_array(image)
            image_array = (
                cp.asarray(
                    image.astype(np.float32, copy=False),
                    dtype=cp.float32,
                )
                if np.issubdtype(image.dtype, np.floating)
                else cp.asarray(
                    image.astype(np.float32) / 255.0,
                    dtype=cp.float32,
                )
            )
            return self._finalize_prepared_array(image_array)
        self._validate_nhwc_rgb_array(image)
        image_array = (
            image.astype(cp.float32, copy=False)
            if cp.issubdtype(image.dtype, cp.floating)
            else image.astype(cp.float32) / 255.0
        )
        return self._finalize_prepared_array(image_array)

    def _validate_nhwc_rgb_array(
        self,
        image: np.ndarray | cp.ndarray,
    ) -> None:
        if image.ndim != 4 or image.shape[-1] != 3:
            msg = "输入数组必须为 NHWC RGB。"
            raise ValueError(msg)
        if image.shape[0] == 0:
            msg = "输入批次不能为空。"
            raise ValueError(msg)

    def _finalize_prepared_array(
        self,
        image_array: cp.ndarray,
    ) -> tuple[cp.ndarray, tuple[int, int]]:
        image_array = cp.ascontiguousarray(image_array)
        size = (image_array.shape[2], image_array.shape[1])
        return image_array, size

    def _build_pass_sizes(
        self,
        source_size: tuple[int, int],
        target_size: tuple[int, int],
    ) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        current_width, current_height = source_size
        target_width, target_height = target_size
        while (current_width, current_height) != target_size:
            next_width = self._next_extent(current_width, target_width)
            next_height = self._next_extent(current_height, target_height)
            sizes.append((next_width, next_height))
            current_width, current_height = next_width, next_height
        return sizes

    def _next_extent(self, current: int, target: int) -> int:
        if current == target:
            return current
        if target > current:
            return min(target, current * 2)
        return max(target, math.ceil(current / 2))

    def _resize_once(
        self,
        image_array: cp.ndarray,
        target_size: tuple[int, int],
        *,
        output: cp.ndarray | None = None,
    ) -> cp.ndarray:
        raise NotImplementedError

    def _require_nhwc_output_array(
        self,
        output: cp.ndarray,
        target_size: tuple[int, int],
        *,
        batch_size: int,
    ) -> cp.ndarray:
        target_width, target_height = target_size
        expected_shape = (batch_size, target_height, target_width, 3)
        self._validate_output_array(output, expected_shape)
        return output

    def _validate_output_array(
        self,
        output: cp.ndarray,
        expected_shape: tuple[int, ...],
    ) -> None:
        if output.shape != expected_shape:
            msg = f"输出数组尺寸必须为 {expected_shape}。"
            raise ValueError(msg)
        if output.dtype != cp.float32:
            msg = "输出数组 dtype 必须为 float32。"
            raise ValueError(msg)
        if not output.flags.c_contiguous:
            msg = "输出数组必须是连续内存。"
            raise ValueError(msg)

    def _as_nhwc_batch_array(self, image_array: cp.ndarray) -> cp.ndarray:
        image_array = cp.ascontiguousarray(image_array, dtype=cp.float32)
        if image_array.ndim != 4 or image_array.shape[3] != 3:
            msg = "输入数组必须为 NHWC RGB。"
            raise ValueError(msg)
        if image_array.shape[0] == 0:
            msg = "输入批次不能为空。"
            raise ValueError(msg)
        return image_array
