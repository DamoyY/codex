#pragma once

#include "adaptive_math.cuh"

struct SampleGrid4x4
{
    Rgb a;
    Rgb b;
    Rgb c;
    Rgb d;
    Rgb e;
    Rgb f;
    Rgb g;
    Rgb h;
    Rgb i;
    Rgb j;
    Rgb k;
    Rgb l;
    Rgb m;
    Rgb n;
    Rgb o;
    Rgb p;
};

__device__ inline Rgb load_rgb(
    const float *src,
    int batch_index,
    int src_width,
    int src_height,
    int x,
    int y)
{
    x = clamp_index(x, src_width - 1);
    y = clamp_index(y, src_height - 1);
    int index = ((batch_index * src_height + y) * src_width + x) * 3;
    return make_rgb(src[index], src[index + 1], src[index + 2]);
}

__device__ inline SampleGrid4x4 load_sample_grid(
    const float *src,
    int batch_index,
    int src_width,
    int src_height,
    int src_x,
    int src_y)
{
    SampleGrid4x4 samples;
    samples.a = load_rgb(src, batch_index, src_width, src_height, src_x - 1, src_y - 1);
    samples.b = load_rgb(src, batch_index, src_width, src_height, src_x, src_y - 1);
    samples.c = load_rgb(src, batch_index, src_width, src_height, src_x + 1, src_y - 1);
    samples.d = load_rgb(src, batch_index, src_width, src_height, src_x + 2, src_y - 1);
    samples.e = load_rgb(src, batch_index, src_width, src_height, src_x - 1, src_y);
    samples.f = load_rgb(src, batch_index, src_width, src_height, src_x, src_y);
    samples.g = load_rgb(src, batch_index, src_width, src_height, src_x + 1, src_y);
    samples.h = load_rgb(src, batch_index, src_width, src_height, src_x + 2, src_y);
    samples.i = load_rgb(src, batch_index, src_width, src_height, src_x - 1, src_y + 1);
    samples.j = load_rgb(src, batch_index, src_width, src_height, src_x, src_y + 1);
    samples.k = load_rgb(src, batch_index, src_width, src_height, src_x + 1, src_y + 1);
    samples.l = load_rgb(src, batch_index, src_width, src_height, src_x + 2, src_y + 1);
    samples.m = load_rgb(src, batch_index, src_width, src_height, src_x - 1, src_y + 2);
    samples.n = load_rgb(src, batch_index, src_width, src_height, src_x, src_y + 2);
    samples.o = load_rgb(src, batch_index, src_width, src_height, src_x + 1, src_y + 2);
    samples.p = load_rgb(src, batch_index, src_width, src_height, src_x + 2, src_y + 2);
    return samples;
}

__device__ inline Rgb bilinear_patch(
    const Rgb &top_left,
    const Rgb &top_right,
    const Rgb &bottom_left,
    const Rgb &bottom_right,
    float x,
    float y)
{
    return lerp_rgb(
        lerp_rgb(top_left, top_right, x),
        lerp_rgb(bottom_left, bottom_right, x),
        y);
}

__device__ inline Rgb average4(
    const Rgb &a,
    const Rgb &b,
    const Rgb &c,
    const Rgb &d)
{
    return scale_rgb(
        add_rgb(add_rgb(a, b), add_rgb(c, d)),
        0.25f);
}

__device__ inline Rgb average16(const SampleGrid4x4 &samples)
{
    Rgb row0 = average4(samples.a, samples.b, samples.c, samples.d);
    Rgb row1 = average4(samples.e, samples.f, samples.g, samples.h);
    Rgb row2 = average4(samples.i, samples.j, samples.k, samples.l);
    Rgb row3 = average4(samples.m, samples.n, samples.o, samples.p);
    return average4(row0, row1, row2, row3);
}
