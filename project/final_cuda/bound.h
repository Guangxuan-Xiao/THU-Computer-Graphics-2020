#ifndef BOUND_H
#define BOUND_H

#include <cuda_runtime.h>

#include "ray.h"
#include "vec3.h"
#define INF (1061109567.f)
class AABB {
   public:
    __device__ AABB() {
        bounds[0] = Vec3(INF, INF, INF);
        bounds[1] = Vec3(-INF, -INF, -INF);
    }

    __device__ void set(const Vec3 &lo, const Vec3 &hi) {
        bounds[0] = lo;
        bounds[1] = hi;
    }

    __device__ void updateBound(const Vec3 &vec) {
        for (int i = 0; i < 3; ++i) {
            bounds[0][i] = bounds[0][i] < vec[i] ? bounds[0][i] : vec[i];
            bounds[1][i] = bounds[1][i] < vec[i] ? vec[i] : bounds[1][i];
        }
    }

    __device__ bool intersect(const Ray &r, float &tmin) const {
        Vec3 o(r.origin()), invdir(1.f / r.direction());
        int sgn[3] = {invdir.x() < 0, invdir.y() < 0, invdir.z() < 0};
        float tmax, tymin, tymax, tzmin, tzmax;
        tmin = (bounds[sgn[0]].x() - o.x()) * invdir.x();
        tmax = (bounds[1 - sgn[0]].x() - o.x()) * invdir.x();
        tymin = (bounds[sgn[1]].y() - o.y()) * invdir.y();
        tymax = (bounds[1 - sgn[1]].y() - o.y()) * invdir.y();
        if ((tmin > tymax) || (tymin > tmax)) return false;
        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;
        tzmin = (bounds[sgn[2]].z() - o.z()) * invdir.z();
        tzmax = (bounds[1 - sgn[2]].z() - o.z()) * invdir.z();
        if ((tmin > tzmax) || (tzmin > tmax)) return false;
        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;
        return true;
    }

   private:
    Vec3 bounds[2];
};

#endif  // !BOUND_H