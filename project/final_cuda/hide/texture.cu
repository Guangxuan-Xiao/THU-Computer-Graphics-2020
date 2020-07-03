#define STB_IMAGE_IMPLEMENTATION
#include <cuda_runtime.h>

#include "texture.h"
__device__ int Texture::getIdx(float u, float v) const {
    int pw = int(u * w + w) % w, ph = int(v * h + h) % h;
    return ph * w * c + pw * c;
}

__device__ Vec3 Texture::getColor(float u, float v) const {
    if (!pic) return Vec3(0, 0, 0);
    int idx = getIdx(u, v);
    return Vec3(pic[idx + 0], pic[idx + 1], pic[idx + 2]) / 255.f;
}

__device__ int Bump::getIdx(float u, float v) const {
    int pw = int(u * w + w) % w, ph = int(v * h + h) % h;
    return ph * w * c + pw * c;
}

__device__ float Bump::getGray(int idx) const {
    return (pic[idx] / 255.f - 0.5f) * 2.0f;
}

__device__ float Bump::getDisturb(float u, float v, float* grad) const {
    if (!pic) return 0;
    float disturb = getGray(getIdx(u, v));
    grad[0] =
        w * (getGray(getIdx(u + du, v)) - getGray(getIdx(u - du, v))) / 2.0f;
    grad[1] =
        h * (getGray(getIdx(u, v + dv)) - getGray(getIdx(u, v - dv))) / 2.0f;
    return disturb;
}