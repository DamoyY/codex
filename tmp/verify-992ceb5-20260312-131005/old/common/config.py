from dataclasses import dataclass
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
MILO_DIR = PROJECT_DIR / "benchmark" / "MILO"
RESULTS_DIR = PROJECT_DIR / "results"
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
RECONSTRUCTION_NAME = "adaptive_reconstruction"
RECONSTRUCTION_DISPLAY_NAME = "Adaptive Reconstruction"
DEVICE_NAMES = {"auto", "cpu", "cuda"}
SCALE_SPACE_NAMES = {"log", "linear"}


@dataclass(frozen=True)
class EvaluationConfig:
    data_dir: Path = PROJECT_DIR.parent / "image_dataset"
    output_dir: Path | None = None
    weights: Path = MILO_DIR / "weights" / "MILO.pth"
    device: str = "auto"
    min_scale: float = 0.0625
    max_scale: float = 16.0
    num_scales: int = 50
    scale_space: str = "log"
    plot_xscale: str = "log"
    base_short_side: int = 256
    recursive: bool = False
    dataset_percentage: float = 20.0
    max_images: int | None = None
    progress_every: int = 50
    dpi: int = 200
    milo_batch_images: int = 100
    max_pending_images: int = 1024
    prefetch_workers: int = 6

    def __post_init__(self) -> None:
        if self.device not in DEVICE_NAMES:
            msg = "device 必须为 auto、cpu 或 cuda。"
            raise ValueError(msg)
        if self.scale_space not in SCALE_SPACE_NAMES:
            msg = "scale_space 必须为 log 或 linear。"
            raise ValueError(msg)
        if self.plot_xscale not in SCALE_SPACE_NAMES:
            msg = "plot_xscale 必须为 log 或 linear。"
            raise ValueError(msg)
        if self.base_short_side < 1:
            msg = "base_short_side 必须大于 0。"
            raise ValueError(msg)
        if not (0.0 < self.dataset_percentage <= 100.0):
            msg = "dataset_percentage 必须在 (0, 100] 范围内。"
            raise ValueError(msg)
        if self.progress_every < 1:
            msg = "progress_every 必须大于 0。"
            raise ValueError(msg)
        if self.dpi < 1:
            msg = "dpi 必须大于 0。"
            raise ValueError(msg)
        if self.milo_batch_images < 1:
            msg = "milo_batch_images 必须大于 0。"
            raise ValueError(msg)
        if self.max_pending_images < 1:
            msg = "max_pending_images 必须大于 0。"
            raise ValueError(msg)
        if self.prefetch_workers < 0:
            msg = "prefetch_workers 不能小于 0。"
            raise ValueError(msg)


CONFIG = EvaluationConfig()
