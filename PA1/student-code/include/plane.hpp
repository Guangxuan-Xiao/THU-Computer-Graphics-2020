#ifndef PLANE_H
#define PLANE_H

#include <vecmath.h>

#include <cmath>

#include "object3d.hpp"

// TODO: Implement Plane representing an infinite plane
// function: ax+by+cz=d
// choose your representation , add more fields and fill in the functions

class Plane : public Object3D {
   public:
    // 构造函数要写
    Plane() : normal(Vector3f::UP), d(0) {}

    Plane(const Vector3f &normal, float d, Material *m)
        : Object3D(m), normal(normal), d(d) {
        assert(normal.length() == 1);
    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f o(r.getOrigin()), dir(r.getDirection());
        dir.normalize();
        float cos = Vector3f::dot(normal, dir);
        // 平行
        if (fabs(cos) < 1e-6) return false;
        // d = n.o + t*n.dir => t = (d-n.o)/(n.dir)
        float t = (d - Vector3f::dot(normal, o)) / Vector3f::dot(normal, dir);
        if (t < tmin || t > h.getT()) return false;
        h.set(t, material, normal);
        return true;
    }

   protected:
    Vector3f normal;
    float d;
};

#endif  // PLANE_H
