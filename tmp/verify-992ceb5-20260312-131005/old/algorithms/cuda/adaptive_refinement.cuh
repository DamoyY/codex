#pragma once

#include "adaptive_sampling.cuh"

struct AdaptiveDirection
{
    float dir_x;
    float dir_y;
    float edge_len;
    float len_x;
    float len_y;
    float lob;
    float clp;
};

struct FilterAccum
{
    Rgb color;
    float weight;
};

__device__ inline void accumulate_direction(
    float *dir_x,
    float *dir_y,
    float *edge_len,
    float pp_x,
    float pp_y,
    int weight_kind,
    float l_a,
    float l_b,
    float l_c,
    float l_d,
    float l_e)
{
    float weight = weight_kind == 0   ? (1.0f - pp_x) * (1.0f - pp_y)
                   : weight_kind == 1 ? pp_x * (1.0f - pp_y)
                   : weight_kind == 2 ? (1.0f - pp_x) * pp_y
                                      : pp_x * pp_y;
    float dc = l_d - l_c;
    float cb = l_c - l_b;
    float len_x = safe_rcp(fmaxf(fabsf(dc), fabsf(cb)));
    float dir_x_value = l_d - l_b;
    *dir_x += dir_x_value * weight;
    len_x = saturate(fabsf(dir_x_value) * len_x);
    *edge_len += len_x * len_x * weight;
    float ec = l_e - l_c;
    float ca = l_c - l_a;
    float len_y = safe_rcp(fmaxf(fabsf(ec), fabsf(ca)));
    float dir_y_value = l_e - l_a;
    *dir_y += dir_y_value * weight;
    len_y = saturate(fabsf(dir_y_value) * len_y);
    *edge_len += len_y * len_y * weight;
}

__device__ inline AdaptiveDirection build_direction(
    const SampleGrid4x4 &samples,
    float pp_x,
    float pp_y,
    float scale_x,
    float scale_y)
{
    float dir_x = 0.0f;
    float dir_y = 0.0f;
    float edge_len = 0.0f;

    float b_l = luma(samples.b);
    float c_l = luma(samples.c);
    float e_l = luma(samples.e);
    float f_l = luma(samples.f);
    float g_l = luma(samples.g);
    float h_l = luma(samples.h);
    float i_l = luma(samples.i);
    float j_l = luma(samples.j);
    float k_l = luma(samples.k);
    float l_l = luma(samples.l);
    float n_l = luma(samples.n);
    float o_l = luma(samples.o);

    accumulate_direction(&dir_x, &dir_y, &edge_len, pp_x, pp_y, 0, b_l, e_l, f_l, g_l, j_l);
    accumulate_direction(&dir_x, &dir_y, &edge_len, pp_x, pp_y, 1, c_l, f_l, g_l, h_l, k_l);
    accumulate_direction(&dir_x, &dir_y, &edge_len, pp_x, pp_y, 2, f_l, i_l, j_l, k_l, n_l);
    accumulate_direction(&dir_x, &dir_y, &edge_len, pp_x, pp_y, 3, g_l, j_l, k_l, l_l, o_l);

    float dir_r = dir_x * dir_x + dir_y * dir_y;
    if (dir_r < 1.0f / 32768.0f)
    {
        dir_x = 1.0f;
        dir_y = 0.0f;
    }
    else
    {
        float dir_scale = rsqrtf(dir_r);
        dir_x *= dir_scale;
        dir_y *= dir_scale;
    }

    edge_len = 0.25f * edge_len * edge_len;
    float stretch = (dir_x * dir_x + dir_y * dir_y) * safe_rcp(fmaxf(fabsf(dir_x), fabsf(dir_y)));
    float len_x = 1.0f + (stretch - 1.0f) * edge_len;
    float len_y = 1.0f - 0.5f * edge_len;
    float lob_bias = scale_x > 1.0f || scale_y > 1.0f
                         ? 0.0f
                         : 0.04f + (0.13f - 0.04f) * saturate((safe_rcp(fminf(scale_x, scale_y)) - 1.0f) * 2.0f);
    float lob = 0.5f + (0.25f - lob_bias - 0.5f) * edge_len;
    float clp = safe_rcp(lob);

    AdaptiveDirection direction = {dir_x, dir_y, edge_len, len_x, len_y, lob, clp};
    return direction;
}

__device__ inline void accumulate_filter_tap(
    FilterAccum *accum,
    float off_x,
    float off_y,
    const AdaptiveDirection &direction,
    const Rgb &sample)
{
    float v_x = (off_x * direction.dir_x + off_y * direction.dir_y) * direction.len_x;
    float v_y = (off_x * (-direction.dir_y) + off_y * direction.dir_x) * direction.len_y;
    float d2 = fminf(v_x * v_x + v_y * v_y, direction.clp);
    float w_b = 2.0f / 5.0f * d2 - 1.0f;
    float w_a = direction.lob * d2 - 1.0f;
    w_b = 25.0f / 16.0f * (w_b * w_b) - (25.0f / 16.0f - 1.0f);
    float weight = w_b * (w_a * w_a);
    accum->color = add_rgb(accum->color, scale_rgb(sample, weight));
    accum->weight += weight;
}

__device__ inline Rgb apply_adaptive_filter(
    const SampleGrid4x4 &samples,
    const AdaptiveDirection &direction,
    float pp_x,
    float pp_y)
{
    FilterAccum accum;
    accum.color = make_rgb(0.0f, 0.0f, 0.0f);
    accum.weight = 0.0f;

    accumulate_filter_tap(&accum, 0.0f - pp_x, -1.0f - pp_y, direction, samples.b);
    accumulate_filter_tap(&accum, 1.0f - pp_x, -1.0f - pp_y, direction, samples.c);
    accumulate_filter_tap(&accum, -1.0f - pp_x, 1.0f - pp_y, direction, samples.i);
    accumulate_filter_tap(&accum, 0.0f - pp_x, 1.0f - pp_y, direction, samples.j);
    accumulate_filter_tap(&accum, 0.0f - pp_x, 0.0f - pp_y, direction, samples.f);
    accumulate_filter_tap(&accum, -1.0f - pp_x, 0.0f - pp_y, direction, samples.e);
    accumulate_filter_tap(&accum, 1.0f - pp_x, 1.0f - pp_y, direction, samples.k);
    accumulate_filter_tap(&accum, 2.0f - pp_x, 1.0f - pp_y, direction, samples.l);
    accumulate_filter_tap(&accum, 2.0f - pp_x, 0.0f - pp_y, direction, samples.h);
    accumulate_filter_tap(&accum, 1.0f - pp_x, 0.0f - pp_y, direction, samples.g);
    accumulate_filter_tap(&accum, 1.0f - pp_x, 2.0f - pp_y, direction, samples.o);
    accumulate_filter_tap(&accum, 0.0f - pp_x, 2.0f - pp_y, direction, samples.n);

    return scale_rgb(accum.color, safe_rcp(accum.weight));
}

__device__ inline float compute_channel_lobe(
    float minimum,
    float maximum,
    float filtered_value)
{
    float hit_min = fminf(minimum, filtered_value) * safe_rcp(4.0f * maximum);
    float hit_max = (1.0f - fmaxf(maximum, filtered_value)) / fminf(-1.0e-6f, 4.0f * minimum - 4.0f);
    return fmaxf(-hit_min, hit_max);
}

__device__ inline Rgb refine_upscale(
    const SampleGrid4x4 &samples,
    const Rgb &filtered,
    const Rgb &minimum,
    const Rgb &maximum,
    float pp_x,
    float pp_y)
{
    Rgb north = bilinear_patch(samples.b, samples.c, samples.f, samples.g, pp_x, pp_y);
    Rgb west = bilinear_patch(samples.e, samples.f, samples.i, samples.j, pp_x, pp_y);
    Rgb east = bilinear_patch(samples.g, samples.h, samples.k, samples.l, pp_x, pp_y);
    Rgb south = bilinear_patch(samples.j, samples.k, samples.n, samples.o, pp_x, pp_y);

    Rgb cross_min = min_rgb4(north, west, east, south);
    Rgb cross_max = max_rgb4(north, west, east, south);

    float lobe_r = compute_channel_lobe(cross_min.r, cross_max.r, filtered.r);
    float lobe_g = compute_channel_lobe(cross_min.g, cross_max.g, filtered.g);
    float lobe_b = compute_channel_lobe(cross_min.b, cross_max.b, filtered.b);

    float sharpen = 0.89f;
    float lobe = fmaxf(-0.125f, fminf(fmaxf(fmaxf(lobe_r, lobe_g), lobe_b), 0.0f)) * sharpen;
    float rcp_lobe = safe_rcp(4.0f * lobe + 1.0f);

    Rgb refined = filtered;
    refined = add_rgb(refined, scale_rgb(north, lobe));
    refined = add_rgb(refined, scale_rgb(west, lobe));
    refined = add_rgb(refined, scale_rgb(east, lobe));
    refined = add_rgb(refined, scale_rgb(south, lobe));
    refined = scale_rgb(refined, rcp_lobe);

    return clamp_rgb(refined, minimum, maximum);
}

__device__ inline Rgb blend_downscale_support(
    const SampleGrid4x4 &samples,
    const Rgb &filtered,
    const Rgb &minimum,
    const Rgb &maximum,
    float scale_x,
    float scale_y,
    float edge_len)
{
    float downscale = fmaxf(scale_x, scale_y);
    float support_mix = saturate((downscale - 4.0f) * 0.25f);
    Rgb support = lerp_rgb(
        average4(samples.f, samples.g, samples.j, samples.k),
        average16(samples),
        support_mix);

    float blend = saturate((downscale - 1.0f) * 0.15f);
    blend *= 1.0f - 0.5f * edge_len;

    return clamp_rgb(lerp_rgb(filtered, support, blend), minimum, maximum);
}
