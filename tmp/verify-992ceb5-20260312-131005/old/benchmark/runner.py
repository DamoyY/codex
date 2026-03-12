from pathlib import Path
from time import perf_counter
from typing import Any, cast

import torch

from common.config import (
    CONFIG,
    RECONSTRUCTION_DISPLAY_NAME,
    RECONSTRUCTION_NAME,
    EvaluationConfig,
)
from common.images import collect_image_paths, iter_prefetched_images
from common.runtime import configure_torch_runtime, resolve_device

from .batches import flush_largest_pending_bucket, flush_pending_images
from .milo_metric import MILOMetric
from .reconstruction import AdaptiveReconstructor, PreparedImage
from .reporting import (
    plot_summary,
    resolve_output_dir,
    summarize_losses,
    write_summary_csv,
)
from .scales import build_scales, split_scales


def run_evaluation(config: EvaluationConfig = CONFIG) -> tuple[Path, Path]:
    device = resolve_device(config.device)
    configure_torch_runtime(device)
    use_cuda_fast_path = device.type == "cuda"
    reconstruction = AdaptiveReconstructor()
    output_dir = resolve_output_dir(config.output_dir)
    scales = build_scales(
        config.min_scale,
        config.max_scale,
        config.num_scales,
        config.scale_space,
    )
    scale_layout = split_scales(scales)
    image_paths = collect_image_paths(
        config.data_dir,
        dataset_percentage=config.dataset_percentage,
        recursive=config.recursive,
        max_images=config.max_images,
    )
    model = MILOMetric(config.weights, device)
    if use_cuda_fast_path:
        model = cast(
            "MILOMetric",
            cast("Any", model).to(memory_format=torch.channels_last),
        )
    losses_by_scale: list[list[torch.Tensor]] = [[] for _ in scales]

    print(f"设备: {device}")
    print(f"图片数量: {len(image_paths)}")
    print(f"数据集启用比例: {config.dataset_percentage}%")
    print(f"缩放倍率数量: {len(scales)}")
    print(f"重建方法: {RECONSTRUCTION_DISPLAY_NAME}")
    print(f"数据目录: {config.data_dir}")
    print(f"输出目录: {output_dir}")

    start_time = perf_counter()
    pending_buckets: dict[tuple[int, int], list[PreparedImage]] = {}
    pending_image_count = 0

    with torch.inference_mode():
        for image_index, (image_path, base_image) in enumerate(
            iter_prefetched_images(
                image_paths,
                prefetch_workers=config.prefetch_workers,
            ),
            start=1,
        ):
            prepared_image = reconstruction.prepare(
                base_image,
                config.base_short_side,
            )
            bucket_images = pending_buckets.setdefault(prepared_image.size, [])
            bucket_images.append(prepared_image)
            pending_image_count += 1

            if len(bucket_images) >= config.milo_batch_images:
                flush_pending_images(
                    bucket_images,
                    reconstruction,
                    model,
                    device,
                    scale_layout,
                    losses_by_scale,
                    channels_last=use_cuda_fast_path,
                )
                pending_image_count -= len(bucket_images)
                del pending_buckets[prepared_image.size]

            while pending_image_count >= config.max_pending_images:
                pending_image_count -= flush_largest_pending_bucket(
                    pending_buckets,
                    reconstruction,
                    model,
                    device,
                    scale_layout,
                    losses_by_scale,
                    channels_last=use_cuda_fast_path,
                )

            if (
                image_index == 1
                or image_index % config.progress_every == 0
                or image_index == len(image_paths)
            ):
                elapsed = perf_counter() - start_time
                images_per_second = image_index / elapsed if elapsed > 0 else 0.0
                remaining = (
                    (len(image_paths) - image_index) / images_per_second
                    if images_per_second > 0
                    else 0.0
                )
                print(
                    f"[{image_index}/{len(image_paths)}] "
                    f"{image_path.name} "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s",
                )

        while pending_buckets:
            pending_image_count -= flush_largest_pending_bucket(
                pending_buckets,
                reconstruction,
                model,
                device,
                scale_layout,
                losses_by_scale,
                channels_last=use_cuda_fast_path,
            )

    summary = summarize_losses(scales, losses_by_scale)
    summary_csv = output_dir / f"{RECONSTRUCTION_NAME}_milo_summary.csv"
    summary_plot = output_dir / f"{RECONSTRUCTION_NAME}_milo_curve.png"

    write_summary_csv(summary_csv, summary)
    plot_summary(
        summary_plot,
        summary,
        config.plot_xscale,
        config.dpi,
        f"{RECONSTRUCTION_DISPLAY_NAME} evaluated by MILO",
        f"Mean MILO loss ({RECONSTRUCTION_DISPLAY_NAME})",
    )

    print(f"结果已保存到: {summary_csv}")
    print(f"图像已保存到: {summary_plot}")
    return summary_csv, summary_plot
