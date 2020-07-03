#ifndef MATERIAL_H
#define MATERIAL_H

#include <glut.h>
#include <vecmath.h>

#include <algorithm>
#include <cassert>
#include <iostream>

#include "hit.hpp"
#include "ray.hpp"

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
        Vector3f N = hit.getNormal(), V = -ray.getDirection().normalized();
        Vector3f Lx = dirToLight.normalized();
        Vector3f Rx = (2 * (Vector3f::dot(Lx, N)) * N - Lx).normalized();
        Vector3f shaded =
            lightColor *
            (diffuseColor * relu(Vector3f::dot(Lx, N)) +
             specularColor * (pow(relu(Vector3f::dot(V, Rx)), shininess)));
        return shaded;
    }

    // For OpenGL, this is fully implemented
    void Use() {
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,
                     Vector4f(diffuseColor, 1.0f));
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,
                     Vector4f(specularColor, 1.0f));
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS,
                     Vector2f(shininess * 4.0, 1.0f));
    }

   protected:
    Vector3f diffuseColor;
    Vector3f specularColor;
    float shininess;
    float relu(float x) { return std::max((float)0, x); }
};

#endif  // MATERIAL_H
