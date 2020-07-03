#ifndef REVSURFACE_HPP
#define REVSURFACE_HPP
// 参数曲面

#include "bound.h"
#include "curve.h"
#include "object.h"
#include "triangle.h"
const int resolution = 10;
const int NEWTON_STEPS = 20;
const float NEWTON_EPS = 1e-4;
class RevSurface : public Object {
    Curve *pCurve;
    AABB aabb;

   public:
    __device__ RevSurface(Curve *pCurve, Material *material)
        : pCurve(pCurve), Object(material) {
        for (int i = 0; i < pCurve->num_controls; ++i) {
            if (pCurve->controls[i].z() != 0.0) {
                printf("Profile of revSurface must be flat on xy plane.\n");
            }
        }
        aabb.set(Vec3(-pCurve->radius, pCurve->ymin - 3, -pCurve->radius),
                 Vec3(pCurve->radius, pCurve->ymax + 3, pCurve->radius));
    }

    __device__ ~RevSurface() { delete pCurve; }

    __device__ inline bool intersect(const Ray &r, float tmin, float tmax,
                                     Hit &rec) const override {
        return newtonIntersect(r, rec, tmin, tmax);
    }

    __device__ bool newtonIntersect(const Ray &r, Hit &h, float tmin,
                                    float tmax) const {
        float t, theta, mu;
        if (!aabb.intersect(r, t) || t > tmax) return false;
        getUV(r, t, theta, mu);
        Vec3 normal;
        if (!newton(r, t, theta, mu, normal)) return false;
        // if (!isnormal(mu) || !isnormal(theta) || !isnormal(t)) return false;
        if (t < tmin || mu < pCurve->range[0] || mu > pCurve->range[1] ||
            t > tmax)
            return false;
        h.set(t, theta / (2.f * (float)M_PI), mu, mat_ptr, normal.normalized());
        return true;
    }

    __device__ bool newton(const Ray &r, float &t, float &theta, float &mu,
                           Vec3 &normal) const {
        Vec3 point, dmu, dtheta;
        for (int i = 0; i < NEWTON_STEPS; ++i) {
            while (theta < 0.0) theta += 2 * M_PI;
            while (theta >= 2 * M_PI) theta -= 2 * M_PI;
            if (mu >= 1) mu = 1.0 - FLT_EPSILON;
            if (mu <= 0) mu = FLT_EPSILON;
            getPoint(theta, mu, point, dtheta, dmu);
            Vec3 f = r.origin() + r.direction() * t - point;
            float dist2 = f.squared_length();
            // cout << "Iter " << i + 1 << " t: " << t
            //      << " theta: " << theta / (2 * M_PI) << " mu: " << mu
            //      << " dist2: " << dist2 << endl;
            normal = cross(dmu, dtheta);
            if (dist2 < NEWTON_EPS) return true;
            float D = dot(r.direction(), normal);
            t -= dot(dmu, cross(dtheta, f)) / D;
            mu -= dot(r.direction(), cross(dtheta, f)) / D;
            theta += dot(r.direction(), cross(dmu, f)) / D;
        }
        return false;
    }

    __device__ void getPoint(const float &theta, const float &mu, Vec3 &pt,
                             Vec3 &dtheta, Vec3 &dmu) const {
        CurvePoint cp = pCurve->getPoint(mu);
        float cos_t = cos(theta), sin_t = sin(theta);
        pt = Vec3(cp.V[0] * cos_t - cp.V[1] * sin_t,
                  cp.V[0] * sin_t + cp.V[1] * cos_t, cp.V[2]);
        dmu = Vec3(cp.T[0] * cos_t - cp.T[1] * sin_t,
                   cp.T[0] * sin_t + cp.T[1] * cos_t, cp.T[2]);
        dtheta = Vec3(-cp.V.x() * sin_t, 0, -cp.V.x() * cos_t);
    }

    __device__ void getUV(const Ray &r, const float &t, float &theta,
                          float &mu) const {
        Vec3 pt(r.origin() + r.direction() * t);
        theta = atan2(-pt.z(), pt.x()) + M_PI;
        mu = (pCurve->ymax - pt.y()) / (pCurve->ymax - pCurve->ymin);
    }
};

#endif  // REVSURFACE_HPP
