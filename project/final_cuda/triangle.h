#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <cfloat>
#include <cmath>
#include <iostream>

#include "object.h"
#include "vec3.h"
class Triangle : public Object {
   public:
    __device__ Triangle() = delete;

    // a b c are three vertex positions of the triangle
    __device__ Triangle(const Vec3& a, const Vec3& b, const Vec3& c,
                        Material* m)
        : Object(m),
          a(a),
          b(b),
          c(c),
          an(VEC3ZERO),
          bn(VEC3ZERO),
          cn(VEC3ZERO) {
        normal = cross((b - a), (c - a)).normalized();
        d = dot(normal, a);
        set = false;
    }

    __device__ bool intersect(const Ray& r, float t_min, float t_max,
                              Hit& rec) const override {
        Vec3 o(r.origin()), dir(r.direction());
        float cos = dot(normal, dir);
        if (cos > 0) return false;
        float t = (d - dot(normal, o)) / cos;
        if (t <= t_min || t > t_max) return false;
        Vec3 p(o + dir * t);
        if (!inTriangle(p)) return false;
        float u, v;
        rec.set(t, u, v, mat_ptr, getNormal(p));
        return true;
    }

    __device__ void setVNorm(const Vec3& anorm, const Vec3& bnorm,
                             const Vec3& cnorm) {
        an = anorm;
        bn = bnorm;
        cn = cnorm;
        set = true;
    }

    Vec3 normal;
    Vec3 a, b, c;
    Vec3 an, bn, cn;
    bool set;
    float d;

   protected:
    __device__ bool inTriangle(const Vec3& p) const {
        float va = dot(cross((b - p), (c - p)), normal),
              vb = dot(cross((c - p), (a - p)), normal),
              vc = dot(cross((a - p), (b - p)), normal);
        return (va >= 0 && vb >= 0 && vc >= 0);
    }

    __device__ Vec3 getNormal(const Vec3& p) const {
        if (!set) return normal;
        Vec3 va = (a - p), vb = (b - p), vc = (c - p);
        float ra = cross(vb, vc).length(), rb = cross(vc, va).length(),
              rc = cross(va, vb).length();
        return (ra * an + rb * bn + rc * cn).normalized();
    }
};

#endif  // TRIANGLE_H
