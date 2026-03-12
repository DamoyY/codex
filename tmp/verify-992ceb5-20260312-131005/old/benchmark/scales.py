from dataclasses import dataclass
from math import isclose

import numpy as np


@dataclass(frozen=True)
class ScaleLayout:
    identity_indices: tuple[int, ...]
    non_identity_indices: tuple[int, ...]
    non_identity_values: tuple[float, ...]


def build_scales(
    min_scale: float,
    max_scale: float,
    num_scales: int,
    scale_space: str,
) -> np.ndarray:
    if min_scale <= 0 or max_scale <= 0:
        msg = "缩放倍率必须大于 0。"
        raise ValueError(msg)
    if min_scale >= max_scale:
        msg = "min-scale 必须小于 max-scale。"
        raise ValueError(msg)
    if num_scales < 2:
        msg = "num-scales 至少需要为 2。"
        raise ValueError(msg)

    scales = (
        np.geomspace(min_scale, max_scale, num_scales)
        if scale_space == "log"
        else np.linspace(min_scale, max_scale, num_scales)
    )
    scales = np.concatenate([scales, np.array([1.0], dtype=np.float64)])
    return np.unique(np.round(scales, 12))


def split_scales(scales: np.ndarray) -> ScaleLayout:
    identity_indices: list[int] = []
    non_identity_indices: list[int] = []
    non_identity_values: list[float] = []

    for scale_index, scale in enumerate(scales):
        scale_value = float(scale)
        if isclose(scale_value, 1.0, rel_tol=0.0, abs_tol=1e-12):
            identity_indices.append(scale_index)
            continue
        non_identity_indices.append(scale_index)
        non_identity_values.append(scale_value)

    return ScaleLayout(
        identity_indices=tuple(identity_indices),
        non_identity_indices=tuple(non_identity_indices),
        non_identity_values=tuple(non_identity_values),
    )
