#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>

#include "ray.h"

class Camera {
   public:
    __device__ Camera(Vec3 center, Vec3 direction, Vec3 up, float angle,
                      float imgW, float imgH, float aperture, float focalLength)
        : imgW(imgW), imgH(imgH) {  // angle is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float half_height = tan(angle / 2.0f);
        float half_width = imgW * half_height / imgH;
        origin = center;
        w = normalized(center - direction);
        u = normalized(cross(up, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focalLength * u -
                            half_height * focalLength * v - focalLength * w;
        horizontal = 2.0f * half_width * focalLength * u;
        vertical = 2.0f * half_height * focalLength * v;
    }
    __device__ Ray get_ray(float s, float t,
                           curandState *local_rand_state) const {
        Vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        Vec3 offset = u * rd.x() + v * rd.y();
        return Ray(origin + offset, lower_left_corner + s * horizontal +
                                        t * vertical - origin - offset);
    }

    __device__ Vec3 random_in_unit_disk(curandState *local_rand_state) const {
        Vec3 p;
        do {
            p = 2.0f * Vec3(curand_uniform(local_rand_state),
                            curand_uniform(local_rand_state), 0) -
                Vec3(1, 1, 0);
        } while (dot(p, p) >= 1.0f);
        return p;
    }

    Vec3 origin;
    Vec3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    float lens_radius, imgW, imgH;
};

#endif
