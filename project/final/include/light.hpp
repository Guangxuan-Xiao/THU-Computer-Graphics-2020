#ifndef LIGHT_H
#define LIGHT_H

#include <cmath>

#include "object3d.hpp"
#include "ray.hpp"
#include "utils.hpp"
#include <vecmath.h>
class Light {
   public:
    Light() = default;

    virtual ~Light() = default;

    virtual void getIllumination(const Vector3f &p, Vector3f &dir,
                                 Vector3f &col) const = 0;
    virtual void type() const = 0;
};

class DirectionalLight : public Light {
   public:
    DirectionalLight() = delete;

    DirectionalLight(const Vector3f &d, const Vector3f &c) {
        direction = d.normalized();
        color = c;
    }

    ~DirectionalLight() override = default;

    ///@param p unsed in this function
    ///@param distanceToLight not well defined because it's not a point light
    void getIllumination(const Vector3f &p, Vector3f &dir,
                         Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = -direction;
        col = color;
    }
    void type() const override {
        std::cout << "This is directional light." << std::endl;
    }

   private:
    Vector3f direction;
    Vector3f color;
};

class PointLight : public Light {
   public:
    PointLight() = delete;

    PointLight(const Vector3f &p, const Vector3f &c) {
        position = p;
        color = c;
    }

    ~PointLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir,
                         Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = (position - p);
        dir = dir / dir.length();
        col = color;
    }

    void type() const override {
        std::cout << "This is point light." << std::endl;
    }

   private:
    Vector3f position;
    Vector3f color;
};

class DiskLight : public Light {
   public:
    DiskLight() = delete;

    DiskLight(const Vector3f &p, const Vector3f &dir, const Vector3f &c,
              float r) {
        position = p;
        color = c;
        direction = dir.normalized();
        radius = r;
        ons(dir, u, v);
    }

    ~DiskLight() override = default;

    void getIllumination(const Vector3f &p, Vector3f &dir,
                         Vector3f &col) const override {
        // the direction to the light is the opposite of the
        // direction of the directional light source
        dir = (position - p);
        dir = dir / dir.length();
        col = color;
    }

    Ray getRay() const {
        float alpha = RND2 * 2 * M_PI;
        Vector3f p = position + cos(alpha) * u + sin(alpha) * v;
        return Ray(p, Matrix3f(u, v, direction) * cosineHemisphere(RND2, RND2));
    }

    void type() const override {
        std::cout << "This is disk light." << std::endl;
    }
    Vector3f position, direction, u, v;
    Vector3f color;
    float radius;
};

#endif  // LIGHT_H
