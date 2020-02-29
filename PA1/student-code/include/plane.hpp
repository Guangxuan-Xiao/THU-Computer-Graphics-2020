#ifndef PLANE_H
#define PLANE_H

#include "object3d.hpp"
#include <vecmath.h>
#include <cmath>

// TODO: Implement Plane representing an infinite plane
// function: ax+by+cz=d
// choose your representation , add more fields and fill in the functions

class Plane : public Object3D {
public:
    // 构造函数要写，求交函数
    Plane() {

    }

    Plane(const Vector3f &normal, float d, Material *m) : Object3D(m) {

    }

    ~Plane() override = default;

    bool intersect(const Ray &r, Hit &h, float tmin) override {
        return false;
    }

protected:


};

#endif //PLANE_H
		

