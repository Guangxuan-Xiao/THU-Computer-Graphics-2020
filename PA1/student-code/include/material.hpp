#ifndef MATERIAL_H
#define MATERIAL_H

#include <vecmath.h>

#include <cassert>
#include <iostream>

#include "hit.hpp"
#include "ray.hpp"

// TODO: Implement Shade function that computes Phong introduced in class.
class Material {
   public:
    explicit Material(const Vector3f &d_color,
                      const Vector3f &s_color = Vector3f::ZERO, float s = 0)
        : diffuseColor(d_color), specularColor(s_color), shininess(s) {}

    virtual ~Material() = default;

    virtual Vector3f getDiffuseColor() const { return diffuseColor; }

    Vector3f Shade(const Ray &ray, const Hit &hit, const Vector3f &dirToLight,
                   const Vector3f &lightColor) {
        // phong模型的公式套进去就可以
        Vector3f N = hit.getNormal(), V = -ray.getDirection();
        Vector3f Rx = 2 * (Vector3f::dot(dirToLight, N)) * N - dirToLight;
        Vector3f shaded =
            lightColor *
            (diffuseColor * relu(Vector3f::dot(dirToLight, N)) +
             specularColor * (pow(relu(Vector3f::dot(V, Rx)), shininess)));
        return shaded;
    }

   protected:
    Vector3f diffuseColor;   // 漫反射系数
    Vector3f specularColor;  // 高光系数
    float shininess;         //高光指数
    float relu(float x) { return std::max((float)0, x); }
};

#endif  // MATERIAL_H
