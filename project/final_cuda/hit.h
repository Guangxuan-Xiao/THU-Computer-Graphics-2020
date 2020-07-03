#ifndef HIT_H
#define HIT_H
#include "vec3.h"
class Material;
class Hit {
   public:
    float t, u, v;
    Vec3 p;
    Vec3 normal;
    Material* mat_ptr;
    __device__ void set(float _t, float _u, float _v, Material* mat,
                        const Vec3& _normal) {
        t = _t;
        u = _u;
        v = _v;
        mat_ptr = mat;
        normal = _normal;
    }
};
#endif  // !HIT_H
