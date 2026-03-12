from __future__ import annotations

import math
from dataclasses import dataclass

import cupy as cp
import numpy as np


@dataclass(frozen=True)
class AxisPlan:
    taps: int
    indices: cp.ndarray
    weights: cp.ndarray


def _pack_axis_plan(
    entries: list[tuple[np.ndarray, np.ndarray]],
    max_taps: int,
) -> AxisPlan:
    dst_length = len(entries)
    indices = np.zeros((dst_length, max_taps), dtype=np.int32)
    weights = np.zeros((dst_length, max_taps), dtype=np.float32)
    for dst_idx, (axis_indices, axis_weights) in enumerate(entries):
        tap_count = axis_indices.shape[0]
        indices[dst_idx, :tap_count] = axis_indices
        weights[dst_idx, :tap_count] = axis_weights
    return AxisPlan(
        taps=max_taps,
        indices=cp.asarray(indices),
        weights=cp.asarray(weights),
    )


def _lanczos3(value: float) -> float:
    value = abs(value)
    if value >= 3.0:
        return 0.0
    if value < 1.0e-6:
        return 1.0
    angle = value * math.pi
    return math.sin(angle) * math.sin(angle / 3.0) * 3.0 / (angle * angle)


def build_lanczos3_axis_plan(src_length: int, dst_length: int) -> AxisPlan:
    scale = src_length / dst_length
    filter_scale = max(scale, 1.0)
    radius = 3.0 * filter_scale
    max_taps = 1
    entries: list[tuple[np.ndarray, np.ndarray]] = []
    for dst_index in range(dst_length):
        src_pos = (dst_index + 0.5) * scale - 0.5
        start = math.floor(src_pos - radius)
        end = math.ceil(src_pos + radius)
        samples: list[tuple[int, float]] = []
        weight_sum = 0.0
        for sample in range(start, end + 1):
            weight = _lanczos3(((sample + 0.5) - src_pos) / filter_scale)
            if weight == 0.0:
                continue
            clamped = min(max(sample, 0), src_length - 1)
            samples.append((clamped, weight))
            weight_sum += weight
        if not samples or abs(weight_sum) < 1.0e-8:
            nearest = min(max(math.floor(src_pos + 0.5), 0), src_length - 1)
            axis_indices = np.array([nearest], dtype=np.int32)
            axis_weights = np.array([1.0], dtype=np.float32)
        else:
            axis_indices = np.fromiter(
                (sample for sample, _ in samples),
                dtype=np.int32,
            )
            axis_weights = np.fromiter(
                (weight / weight_sum for _, weight in samples),
                dtype=np.float32,
            )
        max_taps = max(max_taps, axis_indices.shape[0])
        entries.append((axis_indices, axis_weights))
    return _pack_axis_plan(entries, max_taps)
