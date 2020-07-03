#ifndef RENDER_H
#define RENDER_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include <cfloat>

#include "camera.h"
#include "object.h"
#include "object_list.h"
#include "ray.h"
#include "scene.h"
#include "vec3.h"

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ Vec3 color(const Ray &r, Object **world,
                      curandState *local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation(1.0, 1.0, 1.0), c(0, 0, 0);
    for (int i = 0; i < 20; i++) {
        Hit rec;
        if ((*world)->intersect(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                                 local_rand_state);
            c += cur_attenuation * rec.mat_ptr->emission;
            cur_attenuation *= attenuation;
            cur_ray = scattered;
        } else
            return c;
    }
    return c;  // exceeded recursion
}

__global__ void render(Vec3 *fb, int max_x, int max_y, int ns, Object **world,
                       Camera **cam, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    // printf("%d\n", pixel_index);
    curandState local_rand_state = rand_state[pixel_index];
    Vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    // col[0] = sqrt(col[0]);
    // col[1] = sqrt(col[1]);
    // col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

#endif  // !RENDER_H
