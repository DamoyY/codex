from dataclasses import dataclass

import cupy as cp
from PIL import Image

from algorithms.adaptive_core import AdaptiveUpscaler
from common.config import RECONSTRUCTION_DISPLAY_NAME, RECONSTRUCTION_NAME
from common.resizing import BoxFilterResizer, Lanczos3Resizer


@dataclass(frozen=True)
class PreparedImage:
    image_array: cp.ndarray
    size: tuple[int, int]


def resolve_short_side_size(
    image_size: tuple[int, int],
    short_side: int,
) -> tuple[int, int]:
    width, height = image_size
    current_short_side = min(width, height)
    if current_short_side == short_side:
        return image_size
    scale = short_side / current_short_side
    return max(1, round(width * scale)), max(1, round(height * scale))


def resolve_intermediate_size(
    image_size: tuple[int, int],
    scale: float,
) -> tuple[int, int]:
    width, height = image_size
    ratio = scale if scale < 1.0 else 1.0 / scale
    return max(1, round(width * ratio)), max(1, round(height * ratio))


class AdaptiveReconstructor:
    name = RECONSTRUCTION_NAME
    display_name = RECONSTRUCTION_DISPLAY_NAME

    def __init__(self) -> None:
        self.box_filter = BoxFilterResizer()
        self.lanczos3 = Lanczos3Resizer()
        self.upscaler = AdaptiveUpscaler()

    def _concat_prepared_image_batches(
        self,
        prepared_images: list[PreparedImage],
    ) -> cp.ndarray:
        image_size = prepared_images[0].size
        batch_arrays: list[cp.ndarray] = []
        for prepared_image in prepared_images:
            if prepared_image.size != image_size:
                msg = "批量重建要求所有图片尺寸一致。"
                raise ValueError(msg)
            image_array = cp.ascontiguousarray(
                prepared_image.image_array,
                dtype=cp.float32,
            )
            expected_shape = (1, image_size[1], image_size[0], 3)
            if image_array.shape != expected_shape:
                msg = (
                    f"prepared_image.image_array 尺寸必须为 {expected_shape}。"
                )
                raise ValueError(msg)
            batch_arrays.append(image_array)
        return cp.concatenate(batch_arrays, axis=0)

    def prepare(
        self,
        image: Image.Image,
        short_side: int,
    ) -> PreparedImage:
        target_size = resolve_short_side_size(image.size, short_side)
        if target_size[0] < image.size[0] or target_size[1] < image.size[1]:
            image_array = self.box_filter.resize_array(image, target_size)
        else:
            image_array = self.lanczos3.resize_array(image, target_size)
        return PreparedImage(image_array=image_array, size=target_size)

    def reconstruct_array(
        self,
        prepared_image: PreparedImage,
        scale: float,
        *,
        output: cp.ndarray | None = None,
    ) -> cp.ndarray:
        image_size = prepared_image.size
        image_array = cp.ascontiguousarray(
            prepared_image.image_array,
            dtype=cp.float32,
        )
        if (
            image_array.ndim != 4
            or image_array.shape[1:] != (image_size[1], image_size[0], 3)
            or image_array.shape[0] == 0
        ):
            msg = "prepared_image.image_array 尺寸与 size 不一致。"
            raise ValueError(msg)
        intermediate_size = resolve_intermediate_size(image_size, scale)
        if scale < 1.0:
            scaled_array = self.upscaler.resize_array(
                image_array,
                intermediate_size,
            )
            return self.lanczos3.resize_array(
                scaled_array,
                image_size,
                output=output,
            )

        scaled_array = self.box_filter.resize_array(
            image_array,
            intermediate_size,
        )
        return self.upscaler.resize_array(
            scaled_array,
            image_size,
            output=output,
        )

    def reconstruct_batch_array(
        self,
        prepared_images: list[PreparedImage],
        scale: float,
    ) -> cp.ndarray:
        if not prepared_images:
            msg = "prepared_images 不能为空。"
            raise ValueError(msg)
        batch_input = self._concat_prepared_image_batches(prepared_images)
        return self.reconstruct_array(
            PreparedImage(
                image_array=batch_input,
                size=prepared_images[0].size,
            ),
            scale,
        )
