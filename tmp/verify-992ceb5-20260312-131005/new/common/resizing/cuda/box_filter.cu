extern "C"
{
    __device__ inline int clamp_index(int value, int upper)
    {
        return max(0, min(value, upper));
    }
    __global__ void box_filter_resize_kernel(
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
        float src_x0 = (float)x * scale_x;
        float src_x1 = src_x0 + scale_x;
        float src_y0 = (float)y * scale_y;
        float src_y1 = src_y0 + scale_y;
        int start_x = (int)floorf(src_x0);
        int end_x = (int)ceilf(src_x1) - 1;
        int start_y = (int)floorf(src_y0);
        int end_y = (int)ceilf(src_y1) - 1;
        float acc_r = 0.0f;
        float acc_g = 0.0f;
        float acc_b = 0.0f;
        float acc_w = 0.0f;
        for (int sample_y = start_y; sample_y <= end_y; ++sample_y)
        {
            float overlap_y = fminf(src_y1, (float)sample_y + 1.0f) - fmaxf(src_y0, (float)sample_y);
            if (overlap_y <= 0.0f)
            {
                continue;
            }
            int clamped_y = clamp_index(sample_y, src_height - 1);
            for (int sample_x = start_x; sample_x <= end_x; ++sample_x)
            {
                float overlap_x = fminf(src_x1, (float)sample_x + 1.0f) - fmaxf(src_x0, (float)sample_x);
                if (overlap_x <= 0.0f)
                {
                    continue;
                }
                float weight = overlap_x * overlap_y;
                int clamped_x = clamp_index(sample_x, src_width - 1);
                int index = ((batch * src_height + clamped_y) * src_width + clamped_x) * 3;
                acc_r += src[index] * weight;
                acc_g += src[index + 1] * weight;
                acc_b += src[index + 2] * weight;
                acc_w += weight;
            }
        }
        int out_index = ((batch * dst_height + y) * dst_width + x) * 3;
        if (acc_w <= 0.0f)
        {
            int nearest_x = clamp_index((int)floorf(src_x0), src_width - 1);
            int nearest_y = clamp_index((int)floorf(src_y0), src_height - 1);
            int nearest_index = ((batch * src_height + nearest_y) * src_width + nearest_x) * 3;
            dst[out_index] = src[nearest_index];
            dst[out_index + 1] = src[nearest_index + 1];
            dst[out_index + 2] = src[nearest_index + 2];
            return;
        }
        float inv_w = 1.0f / acc_w;
        dst[out_index] = fminf(fmaxf(acc_r * inv_w, 0.0f), 1.0f);
        dst[out_index + 1] = fminf(fmaxf(acc_g * inv_w, 0.0f), 1.0f);
        dst[out_index + 2] = fminf(fmaxf(acc_b * inv_w, 0.0f), 1.0f);
    }
}
