#ifndef PLANE_H
#define PLANE_H

#include <cfloat>
#include <cmath>

#include "object.h"
#include "vec3.h"
class Plane : public Object {
   public:
    __device__ Plane(const Vec3 &norm, float d, Material *m,
                     Bump *bump = nullptr)
        : Object(m), normal(normalized(norm)), d(d), bump(bump) {
        uaxis = cross(VEC3UP, normal);
    }

    __device__ bool intersect(const Ray &r, float t_min, float t_max,
                              Hit &rec) const override {
        Vec3 o(r.origin()), dir(r.direction());
        dir.normalize();
        float cos = dot(normal, dir);
        if (cos > -FLT_EPSILON) return false;
        // d = n.o + t*n.dir => t = (d-n.o)/(n.dir)
        float t = (d - dot(normal, o)) / cos;
        if (t < t_min || t > t_max) return false;
        float u, v;
        getUV(u, v, o + dir * t);
        rec.set(t, u, v, mat_ptr, getNormal(u, v));
        return true;
    }

    __device__ void getUV(float &u, float &v, const Vec3 &p) const {
        v = p.y();
        u = dot(p - d * normal, uaxis);
    }

    __device__ Vec3 getNormal(float u, float v) const {
        if (!bump) return normal;
        float grad[2] = {0};
        float f = bump->getDisturb(u, v, grad);
        if (fabs(f) < FLT_EPSILON) return normal;
        if (uaxis.squared_length() < FLT_EPSILON) return normal;
        return cross(uaxis + normal * grad[0], VEC3UP + normal * grad[1])
            .normalized();
    }

   protected:
    Vec3 normal, uaxis;
    float d;
    Bump *bump;
};

#endif  // PLANE_H
