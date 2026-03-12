extern "C"
{
    __device__ inline float saturate(float value)
    {
        return fminf(fmaxf(value, 0.0f), 1.0f);
    }
    __global__ void horizontal_resize_kernel(
        const float *src,
        int batch_size,
        int src_height,
        int src_width,
        const int *indices,
        const float *weights,
        int taps,
        float *dst,
        int dst_width,
        int clamp_output)
    {
        int batch = blockIdx.z;
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (batch >= batch_size || x >= dst_width || y >= src_height)
        {
            return;
        }
        int plan_offset = x * taps;
        int out_index = ((batch * src_height + y) * dst_width + x) * 3;
        float acc_r = 0.0f;
        float acc_g = 0.0f;
        float acc_b = 0.0f;
        for (int tap = 0; tap < taps; ++tap)
        {
            float weight = weights[plan_offset + tap];
            if (weight == 0.0f)
            {
                continue;
            }
            int index = ((batch * src_height + y) * src_width + indices[plan_offset + tap]) * 3;
            acc_r += src[index] * weight;
            acc_g += src[index + 1] * weight;
            acc_b += src[index + 2] * weight;
        }
        if (clamp_output != 0)
        {
            acc_r = saturate(acc_r);
            acc_g = saturate(acc_g);
            acc_b = saturate(acc_b);
        }
        dst[out_index] = acc_r;
        dst[out_index + 1] = acc_g;
        dst[out_index + 2] = acc_b;
    }
    __global__ void vertical_resize_kernel(
        const float *src,
        int batch_size,
        int src_height,
        int src_width,
        const int *indices,
        const float *weights,
        int taps,
        float *dst,
        int dst_height,
        int clamp_output)
    {
        int batch = blockIdx.z;
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        if (batch >= batch_size || x >= src_width || y >= dst_height)
        {
            return;
        }
        int plan_offset = y * taps;
        int out_index = ((batch * dst_height + y) * src_width + x) * 3;
        float acc_r = 0.0f;
        float acc_g = 0.0f;
        float acc_b = 0.0f;
        for (int tap = 0; tap < taps; ++tap)
        {
            float weight = weights[plan_offset + tap];
            if (weight == 0.0f)
            {
                continue;
            }
            int index = ((batch * src_height + indices[plan_offset + tap]) * src_width + x) * 3;
            acc_r += src[index] * weight;
            acc_g += src[index + 1] * weight;
            acc_b += src[index + 2] * weight;
        }
        if (clamp_output != 0)
        {
            acc_r = saturate(acc_r);
            acc_g = saturate(acc_g);
            acc_b = saturate(acc_b);
        }
        dst[out_index] = acc_r;
        dst[out_index + 1] = acc_g;
        dst[out_index + 2] = acc_b;
    }
}
