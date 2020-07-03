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
    Plane(const Vector3f &normal, float d, Material *m)
        : Object3D(m), normal(normal.normalized()), d(d) {
        uaxis = Vector3f::cross(Vector3f::UP, normal);
    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h) override {
        Vector3f o(r.getOrigin()), dir(r.getDirection());
        // dir.normalize();
        float cos = Vector3f::dot(normal, dir);
        // 平行
        if (cos > -1e-6) return false;
        // d = n.o + t*n.dir => t = (d-n.o)/(n.dir)
        float t = (d - Vector3f::dot(normal, o)) / cos;
        if (t < 0 || t > h.getT()) return false;
        float u, v;
        Vector3f p(o + dir * t);
        getUV(u, v, p);
        h.set(t, material, getNormal(u, v), material->getColor(u, v), p);
        return true;
    }

    void getUV(float &u, float &v, const Vector3f &p) {
        v = p.y();
        u = Vector3f::dot(p - d * normal, uaxis);
    }

    Vector3f getNormal(float u, float v) {
        Vector2f grad(0);
        float f = material->bump.getDisturb(u, v, grad);
        if (fabs(f) < FLT_EPSILON) return normal;
        if (uaxis.squaredLength() < FLT_EPSILON) return normal;
        return Vector3f::cross(uaxis + normal * grad[0],
                               Vector3f::UP + normal * grad[1])
            .normalized();
    }
    Vector3f min() const override {
        return -INF * Vector3f(fabs(normal.x()) < 1 - FLT_EPSILON,
                               fabs(normal.y()) < 1 - FLT_EPSILON,
                               fabs(normal.z()) < 1 - FLT_EPSILON) +
               normal * d;
    }
    Vector3f max() const override {
        return INF * Vector3f(fabs(normal.x()) < 1 - FLT_EPSILON,
                              fabs(normal.y()) < 1 - FLT_EPSILON,
                              fabs(normal.z()) < 1 - FLT_EPSILON) +
               normal * d;
    }
    Vector3f center() const override { return normal * d; }
    vector<Object3D *> getFaces() override { return {(Object3D *)this}; }

   protected:
    Vector3f normal, uaxis;
    float d;
};

#endif  // PLANE_H
