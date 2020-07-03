#ifndef TRIANGLE_H
#define TRIANGLE_H

#include <vecmath.h>

#include <cfloat>
#include <cmath>
#include <iostream>

#include "object3d.hpp"
#include "utils.hpp"
using namespace std;

// TODO: implement this class and add more fields as necessary,
class Triangle : public Object3D {
   public:
    Triangle() = delete;

    // a b c are three vertex positions of the triangle
    Triangle(const Vector3f& a, const Vector3f& b, const Vector3f& c,
             Material* m)
        : Object3D(m),
          a(a),
          b(b),
          c(c),
          an(Vector3f::ZERO),
          bn(Vector3f::ZERO),
          cn(Vector3f::ZERO) {
        normal = Vector3f::cross((b - a), (c - a)).normalized();
        d = Vector3f::dot(normal, a);
        bound[0] = minE(minE(a, b), c);
        bound[1] = maxE(maxE(a, b), c);
        cen = (a + b + c) / 3;
        nSet = false;
        tSet = false;
    }

    bool intersect(const Ray& r, Hit& h) override {
        Vector3f o(r.getOrigin()), dir(r.getDirection());
        Vector3f v0v1 = b - a;
        Vector3f v0v2 = c - a;
        Vector3f pvec = Vector3f::cross(dir, v0v2);
        float det = Vector3f::dot(v0v1, pvec);
        // IF CULLING
        // if (det < FLT_EPSILON) return false;
        // ray and triangle are parallel if det is close to 0
        if (fabs(det) < FLT_EPSILON) return false;
        float invDet = 1 / det;
        Vector3f tvec = o - a;
        float u = Vector3f::dot(tvec, pvec) * invDet;
        if (u < 0 || u > 1) return false;
        Vector3f qvec = Vector3f::cross(tvec, v0v1);
        float v = Vector3f::dot(dir, qvec) * invDet;
        if (v < 0 || u + v > 1) return false;
        float t = Vector3f::dot(v0v2, qvec) * invDet;
        if (t <= 0 || t > h.getT()) return false;
        Vector3f p(o + dir * t);
        getUV(p, u, v);
        h.set(t, material, getNorm(p), material->getColor(u, v), p);
        return true;
    }

    void setVNorm(const Vector3f& anorm, const Vector3f& bnorm,
                  const Vector3f& cnorm) {
        an = anorm;
        bn = bnorm;
        cn = cnorm;
        nSet = true;
    }

    void setVT(const Vector2f& _at, const Vector2f& _bt, const Vector2f& _ct) {
        at = _at;
        bt = _bt;
        ct = _ct;
        tSet = true;
    }

    Vector3f min() const override { return bound[0]; }
    Vector3f max() const override { return bound[1]; }
    Vector3f center() const override { return cen; }
    vector<Object3D*> getFaces() override { return {(Object3D*)this}; }
    Vector3f normal;
    Vector3f a, b, c, cen;
    Vector2f at, bt, ct;
    Vector3f an, bn, cn;
    Vector3f bound[2];
    float d;
    bool nSet = false;
    bool tSet = false;

   protected:
    bool inTriangle(const Vector3f& p) {
        float va = Vector3f::dot(Vector3f::cross((b - p), (c - p)), normal),
              vb = Vector3f::dot(Vector3f::cross((c - p), (a - p)), normal),
              vc = Vector3f::dot(Vector3f::cross((a - p), (b - p)), normal);
        return (va >= 0 && vb >= 0 && vc >= 0);
    }

    Vector3f getNorm(const Vector3f& p) {
        if (!nSet) return normal;
        Vector3f va = (a - p), vb = (b - p), vc = (c - p);
        float ra = Vector3f::cross(vb, vc).length(),
              rb = Vector3f::cross(vc, va).length(),
              rc = Vector3f::cross(va, vb).length();
        return (ra * an + rb * bn + rc * cn).normalized();
    }

    void getUV(const Vector3f& p, float& u, float& v) {
        if (!tSet) return;
        Vector3f va = (a - p), vb = (b - p), vc = (c - p);
        float ra = Vector3f::cross(vb, vc).length(),
              rb = Vector3f::cross(vc, va).length(),
              rc = Vector3f::cross(va, vb).length();
        Vector2f uv = (ra * at + rb * bt + rc * ct) / (ra + rb + rc);
        u = uv.x();
        v = uv.y();
    }

    Ray randomRay(int axis= -1, long long int seed=0) const override {
        float r1 = random(axis, seed), r2 = random(axis, seed);
        if (r1 + r2 > 1) {
            r1 = 1 - r1;
            r2 = 1 - r2;
        }
        return Ray(r1 * b + r2 * c + (1 - r1 - r2) * a, diffDir(normal, axis, seed));
    }
};

#endif  // TRIANGLE_H
