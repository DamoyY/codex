import torch

from common.runtime import milo_autocast
from common.tensors import (
    cupy_batch_array_to_tensor,
    cupy_images_to_tensor_batch,
)

from .milo_metric import MILOMetric
from .reconstruction import AdaptiveReconstructor, PreparedImage
from .scales import ScaleLayout


def flush_pending_images(
    prepared_images: list[PreparedImage],
    reconstruction: AdaptiveReconstructor,
    model: MILOMetric,
    device: torch.device,
    scale_layout: ScaleLayout,
    losses_by_scale: list[list[torch.Tensor]],
    *,
    channels_last: bool,
) -> None:
    if not prepared_images:
        return
    batch_losses_by_scale: list[list[torch.Tensor]] = [
        [] for _ in losses_by_scale
    ]
    reference_batch: torch.Tensor | None = None
    reference_pyramid: tuple[torch.Tensor, ...] | None = None
    distorted_batch: torch.Tensor | None = None
    distorted_array_batch: object | None = None
    identity_loss_values: torch.Tensor | None = None
    reference_batch = cupy_images_to_tensor_batch(
        [prepared_image.image_array for prepared_image in prepared_images],
        device=device,
        channels_last=channels_last,
    )
    identity_loss_values = reference_batch.new_zeros(
        (len(prepared_images),),
        dtype=torch.float32,
    )
    for scale_index in scale_layout.identity_indices:
        batch_losses_by_scale[scale_index].append(identity_loss_values)
    with milo_autocast(device):
        reference_pyramid = model.build_reference_pyramid(reference_batch)
        for scale_index, scale in zip(
            scale_layout.non_identity_indices,
            scale_layout.non_identity_values,
            strict=False,
        ):
            distorted_array_batch = reconstruction.reconstruct_batch_array(
                prepared_images,
                scale,
            )
            distorted_batch = cupy_batch_array_to_tensor(
                distorted_array_batch,
                device=device,
                channels_last=channels_last,
            )
            batch_losses_by_scale[scale_index].append(
                model.loss_per_image(
                    distorted_batch,
                    reference_batch,
                    ref_scales=reference_pyramid,
                )
                .detach()
                .float(),
            )
            distorted_batch = None
            distorted_array_batch = None
    reference_batch = None
    reference_pyramid = None
    identity_loss_values = None
    for losses, batch_losses in zip(
        losses_by_scale,
        batch_losses_by_scale,
        strict=False,
    ):
        losses.extend(batch_losses)


def flush_largest_pending_bucket(
    pending_buckets: dict[tuple[int, int], list[PreparedImage]],
    reconstruction: AdaptiveReconstructor,
    model: MILOMetric,
    device: torch.device,
    scale_layout: ScaleLayout,
    losses_by_scale: list[list[torch.Tensor]],
    *,
    channels_last: bool,
) -> int:
    bucket_key, bucket_images = max(
        pending_buckets.items(),
        key=lambda item: len(item[1]),
    )
    flush_pending_images(
        bucket_images,
        reconstruction,
        model,
        device,
        scale_layout,
        losses_by_scale,
        channels_last=channels_last,
    )
    del pending_buckets[bucket_key]
    return len(bucket_images)
