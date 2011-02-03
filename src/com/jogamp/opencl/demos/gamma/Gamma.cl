    /**
     * gamma correction kernel
     */
    kernel void gamma(global float* image, const float gamma, const float scale, const int max) {
        int index = get_global_id(0);
        if (index >= max)  {
            return;
        }
        image[index] = pow(image[index], gamma) * scale;
    }