#ifndef SPHERE_H
#define SPHERE_H

#include <vecmath.h>

#include <cmath>

#include "object3d.hpp"

// TODO: Implement functions and add more fields as necessary

class Sphere : public Object3D {
   public:
    Sphere() : radius(0), center(0, 0, 0) {
        // unit ball at the center
    }

    Sphere(const Vector3f &center, float radius, Material *material)
        : Object3D(material), center(center), radius(radius) {
        //
    }

    ~Sphere() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        Vector3f o(r.getOrigin()), dir(r.getDirection());
        Vector3f OC(center - o);
        dir.normalize();
        // 计算OC在射线方向的投影长度OH: OC@OH/|OH|
        float OH = Vector3f::dot(OC, dir);
        // 计算CH的长度
        float CH = sqrt(fabs(OC.squaredLength() - OH * OH));
        if (CH > radius) return false;
        // 计算PH的长度
        float PH = sqrt(fabs(radius * radius - CH * CH));
        float t = OH - PH;
        if (t < tmin || t > h.getT()) return false;
        Vector3f P(o + dir * t);
        Vector3f normal((P - center).normalized());
        h.set(t, material, normal);
        return true;
    }

   protected:
    Vector3f center;
    float radius;
};

#endif
