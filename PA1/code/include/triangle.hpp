#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vecmath.h>

#include <cmath>
#include <iostream>

#include "object3d.hpp"
using namespace std;

// TODO: implement this class and add more fields as necessary,
class Triangle : public Object3D {
   public:
    Triangle() = delete;

    // a b c are three vertex positions of the triangle
    Triangle(const Vector3f& a, const Vector3f& b, const Vector3f& c,
             Material* m)
        : Object3D(m), a(a), b(b), c(c) {
        normal = Vector3f::cross((b - a), (c - a)).normalized();
    }

    bool intersect(const Ray& r, Hit& h, float tmin) override {
        Vector3f o(r.getOrigin()), dir(r.getDirection());
        // dir.normalize();
        float cos = Vector3f::dot(normal, dir);
        // 平行
        if (fabs(cos)<1e-6) return false;
        // d = n.o + t*n.dir => t = (d-n.o)/(n.dir)
        float d = Vector3f::dot(normal, a);
        float t = (d - Vector3f::dot(normal, o)) / Vector3f::dot(normal, dir);
        if (t < tmin || t > h.getT()) return false;
        Vector3f p(o + dir * t);
        if (!inTriangle(p)) return false;
        h.set(t, material, normal);
        return true;
    }

    Vector3f normal;
    Vector3f a, b, c;

   protected:
    bool inTriangle(const Vector3f& p) {
        return Vector3f::dot(Vector3f::cross((b - p), (c - p)), normal) >= -1e-6 &&
               Vector3f::dot(Vector3f::cross((c - p), (a - p)), normal) >= -1e-6 &&
               Vector3f::dot(Vector3f::cross((a - p), (b - p)), normal) >= -1e-6;
    }
};

#endif  // TRIANGLE_H
