#pragma once

struct Rgb
{
    float r;
    float g;
    float b;
};

__device__ inline Rgb make_rgb(float r, float g, float b)
{
    Rgb value = {r, g, b};
    return value;
}

__device__ inline int clamp_index(int value, int upper)
{
    return max(0, min(value, upper));
}

__device__ inline float saturate(float value)
{
    return fminf(fmaxf(value, 0.0f), 1.0f);
}

__device__ inline float safe_rcp(float value)
{
    return value > 0.0f ? 1.0f / value : 0.0f;
}

__device__ inline float luma(const Rgb &value)
{
    return (value.r + value.b) * 0.5f + value.g;
}

__device__ inline Rgb add_rgb(const Rgb &left, const Rgb &right)
{
    return make_rgb(
        left.r + right.r,
        left.g + right.g,
        left.b + right.b);
}

__device__ inline Rgb scale_rgb(const Rgb &value, float scale)
{
    return make_rgb(
        value.r * scale,
        value.g * scale,
        value.b * scale);
}

__device__ inline Rgb lerp_rgb(const Rgb &start, const Rgb &end, float t)
{
    return make_rgb(
        start.r + (end.r - start.r) * t,
        start.g + (end.g - start.g) * t,
        start.b + (end.b - start.b) * t);
}

__device__ inline Rgb clamp_rgb(
    const Rgb &value,
    const Rgb &minimum,
    const Rgb &maximum)
{
    return make_rgb(
        fminf(maximum.r, fmaxf(minimum.r, value.r)),
        fminf(maximum.g, fmaxf(minimum.g, value.g)),
        fminf(maximum.b, fmaxf(minimum.b, value.b)));
}

__device__ inline Rgb saturate_rgb(const Rgb &value)
{
    return make_rgb(
        saturate(value.r),
        saturate(value.g),
        saturate(value.b));
}

__device__ inline Rgb min_rgb4(
    const Rgb &a,
    const Rgb &b,
    const Rgb &c,
    const Rgb &d)
{
    return make_rgb(
        fminf(fminf(a.r, b.r), fminf(c.r, d.r)),
        fminf(fminf(a.g, b.g), fminf(c.g, d.g)),
        fminf(fminf(a.b, b.b), fminf(c.b, d.b)));
}

__device__ inline Rgb max_rgb4(
    const Rgb &a,
    const Rgb &b,
    const Rgb &c,
    const Rgb &d)
{
    return make_rgb(
        fmaxf(fmaxf(a.r, b.r), fmaxf(c.r, d.r)),
        fmaxf(fmaxf(a.g, b.g), fmaxf(c.g, d.g)),
        fmaxf(fmaxf(a.b, b.b), fmaxf(c.b, d.b)));
}

__device__ inline void write_rgb(float *dst, int index, const Rgb &value)
{
    dst[index] = value.r;
    dst[index + 1] = value.g;
    dst[index + 2] = value.b;
}
