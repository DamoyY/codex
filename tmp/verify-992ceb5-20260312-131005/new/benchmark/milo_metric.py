import importlib.util
from functools import lru_cache
from math import floor
from pathlib import Path
from types import ModuleType

import torch
from torch.nn import functional

from common.config import MILO_DIR

ScalePyramid = tuple[torch.Tensor, ...]


@lru_cache(maxsize=1)
def load_milo_runner_module() -> ModuleType:
    module_path = MILO_DIR / "MILO_runner.py"
    module_spec = importlib.util.spec_from_file_location(
        "milo_runner_module",
        module_path,
    )
    if module_spec is None or module_spec.loader is None:
        msg = f"无法加载模块: {module_path}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_metric_state_dict(
    module: torch.nn.Module,
    weights_path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loaded_state_dict = torch.load(weights_path, map_location=device)
    module_state_dict = module.state_dict()
    missing_keys = tuple(
        key for key in module_state_dict if key not in loaded_state_dict
    )
    if missing_keys:
        missing_keys_text = ", ".join(missing_keys)
        msg = f"权重文件缺少以下参数: {missing_keys_text}"
        raise RuntimeError(msg)
    return {key: loaded_state_dict[key] for key in module_state_dict}


def build_scale_pyramid(
    image: torch.Tensor,
    number_of_scales: int,
) -> ScalePyramid:
    scales = [image]
    for _ in range(number_of_scales):
        scales.insert(
            0,
            functional.avg_pool2d(
                input=scales[0],
                kernel_size=2,
                stride=2,
                count_include_pad=False,
            ),
        )
    return tuple(scales)


def pad_mask_to_match(
    mask: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pad_bottom = max(0, target.shape[2] - mask.shape[2])
    pad_right = max(0, target.shape[3] - mask.shape[3])
    if pad_bottom == 0 and pad_right == 0:
        return mask
    return functional.pad(
        input=mask,
        pad=[0, pad_right, 0, pad_bottom],
        mode="replicate",
    )


class MILOMetric(torch.nn.Module):
    def __init__(self, weights_path: Path, device: torch.device) -> None:
        super().__init__()
        milo_runner_module = load_milo_runner_module()
        self.mask_finder_1 = milo_runner_module.MaskFinder(7).to(device)
        self.number_of_scales = 3
        state_dict = load_metric_state_dict(self, weights_path, device)
        self.load_state_dict(state_dict, strict=True)
        self.requires_grad_(requires_grad=False)
        self.eval()

    def build_reference_pyramid(self, ref: torch.Tensor) -> ScalePyramid:
        return build_scale_pyramid(ref, self.number_of_scales)

    def mask_generator_from_pyramids(
        self,
        ref_scales: ScalePyramid,
        dist_scales: ScalePyramid,
    ) -> torch.Tensor:
        if len(ref_scales) != len(dist_scales):
            msg = "参考金字塔与失真金字塔层数不一致。"
            raise ValueError(msg)
        smallest_ref = ref_scales[0]
        mask = smallest_ref.new_zeros(
            [
                smallest_ref.shape[0],
                1,
                floor(smallest_ref.shape[2] / 2.0),
                floor(smallest_ref.shape[3] / 2.0),
            ],
        )
        for ref_scale, dist_scale in zip(
            ref_scales,
            dist_scales,
            strict=False,
        ):
            upsampled_mask = functional.interpolate(
                input=mask,
                scale_factor=2,
                mode="bilinear",
                align_corners=True,
            )
            upsampled_mask = pad_mask_to_match(upsampled_mask, ref_scale)
            mask = (
                self.mask_finder_1(
                    torch.cat([ref_scale, dist_scale, upsampled_mask], dim=1),
                )
                + upsampled_mask
            )
        return mask

    def forward(self, dist: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return self.loss_per_image(dist, ref).mean()

    def loss_per_image(
        self,
        dist: torch.Tensor,
        ref: torch.Tensor,
        *,
        ref_scales: ScalePyramid | None = None,
    ) -> torch.Tensor:
        dist_scales = build_scale_pyramid(dist, self.number_of_scales)
        mask = self.mask_generator_from_pyramids(
            self.build_reference_pyramid(ref)
            if ref_scales is None
            else ref_scales,
            dist_scales,
        )
        return (mask * torch.abs(ref - dist)).mean(dim=(1, 2, 3))
