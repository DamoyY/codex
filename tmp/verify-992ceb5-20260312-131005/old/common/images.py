from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from math import ceil
from pathlib import Path

from PIL import Image, ImageOps

from .config import VALID_SUFFIXES


def collect_image_paths(
    data_dir: Path,
    *,
    dataset_percentage: float,
    recursive: bool,
    max_images: int | None,
) -> list[Path]:
    if not data_dir.exists():
        msg = f"数据目录不存在: {data_dir}"
        raise FileNotFoundError(msg)
    iterator = data_dir.rglob("*") if recursive else data_dir.glob("*")
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )
    if not paths:
        msg = f"在 {data_dir} 中没有找到支持的图片文件。"
        raise FileNotFoundError(msg)
    if dataset_percentage < 100.0:
        selected_count = max(
            1,
            ceil(len(paths) * (dataset_percentage / 100.0)),
        )
        paths = paths[:selected_count]
    if max_images is None:
        return paths
    return paths[:max_images]


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def iter_prefetched_images(
    image_paths: list[Path],
    *,
    prefetch_workers: int,
) -> Iterator[tuple[Path, Image.Image]]:
    if prefetch_workers <= 0 or len(image_paths) <= 1:
        for image_path in image_paths:
            yield image_path, load_rgb_image(image_path)
        return
    initial_count = min(len(image_paths), prefetch_workers + 1)
    pending_paths = image_paths[initial_count:]
    with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
        future_queue: deque[tuple[Path, Future[Image.Image]]] = deque(
            (
                image_path,
                executor.submit(load_rgb_image, image_path),
            )
            for image_path in image_paths[:initial_count]
        )
        for image_path in pending_paths:
            current_path, future = future_queue.popleft()
            future_queue.append(
                (
                    image_path,
                    executor.submit(load_rgb_image, image_path),
                ),
            )
            yield current_path, future.result()
        while future_queue:
            current_path, future = future_queue.popleft()
            yield current_path, future.result()
