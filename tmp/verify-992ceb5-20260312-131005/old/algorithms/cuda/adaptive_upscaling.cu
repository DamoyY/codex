#include "adaptive_math.cuh"
#include "adaptive_sampling.cuh"
#include "adaptive_refinement.cuh"

extern "C" __global__ void adaptive_upscaling_kernel(
    const float *src,
    int batch_size,
    int src_height,
    int src_width,
    float *dst,
    int dst_height,
    int dst_width)
{
    int batch = blockIdx.z;
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (batch >= batch_size || x >= dst_width || y >= dst_height)
    {
        return;
    }

    float scale_x = (float)src_width / (float)dst_width;
    float scale_y = (float)src_height / (float)dst_height;
    bool is_upscaling = scale_x < 1.0f || scale_y < 1.0f;
    bool is_downscaling = scale_x > 1.0f || scale_y > 1.0f;

    float pp_x_full = (float)x * scale_x + (0.5f * scale_x - 0.5f);
    float pp_y_full = (float)y * scale_y + (0.5f * scale_y - 0.5f);
    float fp_x = floorf(pp_x_full);
    float fp_y = floorf(pp_y_full);
    float pp_x = pp_x_full - fp_x;
    float pp_y = pp_y_full - fp_y;
    int src_x = (int)fp_x;
    int src_y = (int)fp_y;

    SampleGrid4x4 samples = load_sample_grid(
        src,
        batch,
        src_width,
        src_height,
        src_x,
        src_y);
    AdaptiveDirection direction = build_direction(
        samples,
        pp_x,
        pp_y,
        scale_x,
        scale_y);
    Rgb minimum = min_rgb4(samples.f, samples.g, samples.j, samples.k);
    Rgb maximum = max_rgb4(samples.f, samples.g, samples.j, samples.k);
    Rgb output = apply_adaptive_filter(samples, direction, pp_x, pp_y);

    if (is_upscaling)
    {
        output = refine_upscale(
            samples,
            output,
            minimum,
            maximum,
            pp_x,
            pp_y);
        if (direction.edge_len > 0.64f)
        {
            output = clamp_rgb(output, minimum, maximum);
        }
    }

    if (is_downscaling)
    {
        output = blend_downscale_support(
            samples,
            output,
            minimum,
            maximum,
            scale_x,
            scale_y,
            direction.edge_len);
    }

    int out_index = ((batch * dst_height + y) * dst_width + x) * 3;
    write_rgb(dst, out_index, saturate_rgb(output));
}
